from __future__ import print_function
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import metrics

# --- MODELS ---
from models.lenet5 import LeNet5
#from models.vgg16 import VGG16
#from models.alexnet import AlexNet
#from models.resnext50 import ResNeXt

# Compiles all the training images used into one image
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/lenet5_img.jpeg")

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        torch.cuda.empty_cache()
        output = model(data)
        #output = output.reshape(-1, 369) # used for VGG16 and AlexNet
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, epoch, accuracy_epoch, loss_epoch, y_pred, y_true): #y_true

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Collecting values for the accuracy graph
            accuracy_epoch[0, epoch - 1] = 100. * correct / len(test_loader.dataset)
            accuracy_epoch[1, epoch - 1] = epoch

            # Collecting values for the loss graph
            loss_epoch[0, epoch - 1] = test_loss
            loss_epoch[1, epoch - 1] = epoch

            # Collecting values for the Confusion Matrix
            y_pred.extend(pred)
            y_true.extend(target.view_as(pred))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    epoches = 5
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    save_model = True

    # Check whether CUDA can be used
    use_cuda = torch.cuda.is_available()
    print(use_cuda, 'use cuda')
    # Use CUDA if possible
    device = torch.device("cuda" if use_cuda else "cpu")

    # --- LOADING THE DATA (training, testing) ---
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomResizedCrop(32),
                           transforms.ToTensor()
                       ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Mnist', train=False, transform=transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.ToTensor()
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    # Collect random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    img = torchvision.utils.make_grid(images)
    imsave(img)

    # Build network and run
    model = LeNet5().to(device, dtype=torch.float).to(device)# model = VGG16().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Initialising arrays for graphs and confusion matrix
    accuracy_epoch = np.empty([2, epoches])
    loss_epoch = np.empty([2, epoches])
    y_true = []
    y_pred = []

    for epoch in range(1, epoches + 1):
        torch.cuda.empty_cache()
        train(log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch, accuracy_epoch, loss_epoch, y_pred, y_true) #, y_true
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "./results/mnist_LeNet5.pt")

    #--- GRAPHS ---
    # Accuracy Graph
    graph_accuracy = plt.figure()
    plt.plot(accuracy_epoch[1], accuracy_epoch[0], color='blue')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')

    graph_accuracy.show()
    graph_accuracy.savefig('results/accuracy_LeNet5.jpeg')

    # Loss Graph
    graph_loss = plt.figure()
    plt.plot(loss_epoch[1], loss_epoch[0], color='blue')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')

    graph_loss.show()
    graph_loss.savefig('results/train_loss.jpeg')

    # --- CONFUSION MATRIX ---
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # Print the confusion matrix
    print(metrics.confusion_matrix(y_true, y_pred))
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_true, y_pred, target_names=classes))

if __name__ == '__main__':
    main()

