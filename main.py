from __future__ import print_function
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from models.lenet5 import LeNet5
from models.vgg16 import VGG16
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import hasy_tools

# functions to show an image
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")

def train_vgg16(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        torch.cuda.empty_cache()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = torch.squeeze(data)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    epoches = 10 #14
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    save_model = True

    # Check whether CUDA can be used
    use_cuda = torch.cuda.is_available()
    print(use_cuda, 'use cuda')
    # Use CUDA if possible
    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {} #pin_memory : True

    # --- Load the data ---
    train_data = torchvision.datasets.ImageFolder(root='./ImageFolder/train', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=18, shuffle=True, **kwargs) #151410
    test_data = torchvision.datasets.ImageFolder(root='./ImageFolder/test', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=True, **kwargs) #16823

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    img = torchvision.utils.make_grid(images)
    imsave(img)

    # Build network and run
    model = VGG16().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epoches + 1):
        torch.cuda.empty_cache()
        train_vgg16(log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if save_model:
        print('after save_model')
        torch.save(model.state_dict(), "./results/hasyv2_vgg16.pt")

if __name__ == '__main__':
    main()