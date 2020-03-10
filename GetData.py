
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class GetData():
    def importDataset():
        SEED = 1
        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)
        # For reproducibility
        device = torch.device("cuda" if cuda else "cpu")
        
        torch.manual_seed(SEED)
        
        if cuda:
            torch.cuda.manual_seed(SEED)

        train_transforms = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                       ])
        test_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transforms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transforms)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return trainloader, testloader, classes, device

