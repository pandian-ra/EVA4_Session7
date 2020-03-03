
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



class GetData():
    def importDataset();
        SEED = 1
        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)
        # For reproducibility
        torch.manual_seed(SEED)
        
        if cuda:
            torch.cuda.manual_seed(SEED)

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return (trainloader, testloader, classes)
#
#
#def imshow(img):
#    img = img / 2 + 0.5     # unnormalize
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
#
## get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#
## show images
#imshow(torchvision.utils.make_grid(images))
## print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
#"""2. Define a Convolution Neural Network
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Copy the neural network from the Neural Networks section before and modify it to
#take 3-channel images (instead of 1-channel images as it was defined).
#"""
#
#import torch.nn as nn
#import torch.nn.functional as F
#
#dropout_value = 0.10
#
#class depthwise_separable_conv(nn.Module):
#    def __init__(self, nin, nout):
#        super(depthwise_separable_conv, self).__init__()
#        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
#        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
#
#    def forward(self, x):
#        out = self.depthwise(x)
#        out = self.pointwise(out)
#        return out
#
#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.convblock1 = nn.Sequential(
#            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#            nn.Dropout(dropout_value), # output_size = 30
#            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.Dropout(dropout_value), # output_size = 28
#            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.Dropout(dropout_value), # output_size = 26
#            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.Dropout(dropout_value), # output_size = 24
#            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
#            nn.Dropout(dropout_value) # output_size = 22
#        ) # output_size = 24
#
#        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 10
#        self.transblock1 = nn.Sequential(
#            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
#        ) # output_size = 10
#
#        self.convblock2 = nn.Sequential(
#            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=2, bias=False),
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#            nn.Dropout(dropout_value), # output_size = 12
#            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=2, bias=False),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.Dropout(dropout_value), # output_size = 14
#            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=2, bias=False),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.Dropout(dropout_value), # output_size = 16
#            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=2, bias=False),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
#            nn.Dropout(dropout_value) # output_size = 18
#        ) # output_size = 4
#        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 9
#        self.transblock2 = nn.Sequential(
#            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
#        ) # output_size = 2
#        self.convblock3 = nn.Sequential(
#            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride = 1, dilation =(2,2), padding=0, bias=False),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.Dropout(dropout_value), # output_size = 9
#            nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
#            nn.BatchNorm2d(128),
#            nn.ReLU(),
#            nn.Dropout(dropout_value), # output_size = 7
#         ) # output_size = 10
#        self.pool3 = nn.MaxPool2d(2, 2) # output_size = 3
#        self.transblock3 = nn.Sequential(
#            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
#        ) # output_size = 3
#        self.sepaConv1 = depthwise_separable_conv(32,32)
#        self.b13 = nn.BatchNorm2d(32) # output_size = 3
#        self.convblock4 = nn.Sequential(
#            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
#            nn.BatchNorm2d(64),
#            nn.ReLU(),
#            nn.Dropout(dropout_value), # output_size = 3
#            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=2, bias=False), # output_size = 5
#           )
#
#
#
#        self.gap = nn.Sequential(
#            nn.AvgPool2d(kernel_size=4)
#        ) # output_size = 1
#
#        # SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
#        self.dropout = nn.Dropout(dropout_value)
#        self.convblock8 = nn.Sequential(
#            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
#            # nn.BatchNorm2d(10),
#            # nn.ReLU(),
#            # nn.Dropout(dropout_value)
#        )
#        # self.conv1 = nn.Conv2d(3, 6, 5)
#        # self.pool = nn.MaxPool2d(2, 2)
#        # self.conv2 = nn.Conv2d(6, 16, 5)
#        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
#        # self.fc2 = nn.Linear(120, 84)
#        # self.fc3 = nn.Linear(84, 10)
#
#    def forward(self, x):
#        x = self.convblock1(x)
#        x = self.pool1(x)
#        x = self.transblock1(x)
#
#        x = self.convblock2(x)
#        x = self.pool2(x)
#        x = self.transblock2(x)
#
#        x = self.convblock3(x)
#        x = self.pool3(x)
#        x = self.transblock3(x)
#
#        x = self.sepaConv1(x)
#        x = self.b13(x)
#
#
#        x = self.convblock4(x)
#
#        x = self.gap(x)
#        x = self.convblock8(x)
#
#        x = x.view(-1, 10)
#        return F.log_softmax(x, dim=-1)
#
#model = Net().to(device)
#summary(model, input_size=(3, 32, 32))
#
#!pip install torchsummary
#from torchsummary import summary
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
#print(device)
#model = Net().to(device)
#summary(model, input_size=(3, 32, 32))
#
#"""3. Define a Loss function and optimizer
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Let's use a Classification Cross-Entropy loss and SGD with momentum.
#"""
#
#import torch.optim as optim
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
#
#"""4. Train the network
#^^^^^^^^^^^^^^^^^^^^
#
#This is when things start to get interesting.
#We simply have to loop over our data iterator, and feed the inputs to the
#network and optimize.
#"""
#
#from tqdm import tqdm
#
#train_losses = []
#test_losses = []
#train_acc = []
#test_acc = []
## LAMDA = 0.001
#def train(model, device, train_loader, optimizer, epoch, LAMDA):
#  model.train()
#  pbar = tqdm(train_loader)
#  correct = 0
#  processed = 0
#  criterion= nn.CrossEntropyLoss().to(device)
#
#  for batch_idx, (data, target) in enumerate(pbar):
#    # get samples
#    data, target = data.to(device), target.to(device)
#
#    # Init
#    optimizer.zero_grad()
#    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
#    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
#
#    # Predict
#    y_pred = model(data)
#
#    # Calculate loss
#    regularization_loss = 0
#    for param in model.parameters():
#        regularization_loss += torch.sum(abs(param))
#
#    classify_loss = criterion(y_pred,target)
#    # loss = F.nll_loss(y_pred, target)
#    loss = classify_loss + LAMDA * regularization_loss
#
#    train_losses.append(loss)
#
#    # Backpropagation
#    loss.backward()
#    optimizer.step()
#
#    # Update pbar-tqdm
#
#    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#    correct += pred.eq(target.view_as(pred)).sum().item()
#    processed += len(data)
#
#    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
#    train_acc.append(100*correct/processed)
#
#def test(model, device, test_loader):
#    model.eval()
#    test_loss = 0
#    correct = 0
#    with torch.no_grad():
#        for data, target in test_loader:
#            data, target = data.to(device), target.to(device)
#            output = model(data)
#            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#            correct += pred.eq(target.view_as(pred)).sum().item()
#
#    test_loss /= len(test_loader.dataset)
#    test_losses.append(test_loss)
#
#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#        test_loss, correct, len(test_loader.dataset),
#        100. * correct / len(test_loader.dataset)))
#
#    test_acc.append(100. * correct / len(test_loader.dataset))
#
#model =  Net().to(device)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#EPOCHS = 20
#for epoch in range(EPOCHS):
#    print("EPOCH:", epoch)
#    train(model, device, trainloader, optimizer, epoch, 0.0001)
#    # scheduler.step()
#    test(model, device, testloader)
#
#for epoch in range(10):  # loop over the dataset multiple times
#
#    running_loss = 0.0
#    for i, data in enumerate(trainloader, 0):
#        # get the inputs
#        inputs, labels = data
#        inputs, labels = inputs.to(device), labels.to(device)
#        # zero the parameter gradients
#        optimizer.zero_grad()
#
#        # forward + backward + optimize
#        outputs = model(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#
#        # print statistics
#        running_loss += loss.item()
#        if i % 2000 == 1999:    # print every 2000 mini-batches
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 2000))
#            running_loss = 0.0
#
#print('Finished Training')
#
#
#
#"""5. Test the network on the test data
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#We have trained the network for 2 passes over the training dataset.
#But we need to check if the network has learnt anything at all.
#
#We will check this by predicting the class label that the neural network
#outputs, and checking it against the ground-truth. If the prediction is
#correct, we add the sample to the list of correct predictions.
#
#Okay, first step. Let us display an image from the test set to get familiar.
#"""
#
#dataiter = iter(testloader)
#images, labels = dataiter.next()
#images, labels = images.to(device), labels.to(device)
#
## print images
## imshow(torchvision.utils.make_grid(images))
## print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
#"""Okay, now let us see what the neural network thinks these examples above are:"""
#
#outputs = model(images)
#
#"""The outputs are energies for the 10 classes.
#Higher the energy for a class, the more the network
#thinks that the image is of the particular class.
#So, let's get the index of the highest energy:
#"""
#
#_, predicted = torch.max(outputs, 1)
#
#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(4)))
#
#"""The results seem pretty good.
#
#Let us look at how the network performs on the whole dataset.
#"""
#
#correct = 0
#total = 0
#with torch.no_grad():
#    for data in testloader:
#        images, labels = data
#        images, labels = images.to(device), labels.to(device)
#        outputs = model(images)
#        _, predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()
#
#print('Accuracy of the network on the 10000 test images: %d %%' % (
#    100 * correct / total))
#
#"""That looks waaay better than chance, which is 10% accuracy (randomly picking
#a class out of 10 classes).
#Seems like the network learnt something.
#
#Hmmm, what are the classes that performed well, and the classes that did
#not perform well:
#"""
#
#class_correct = list(0. for i in range(10))
#class_total = list(0. for i in range(10))
#with torch.no_grad():
#    for data in testloader:
#        images, labels = data
#        images, labels = images.to(device), labels.to(device)
#        outputs = model(images)
#        _, predicted = torch.max(outputs, 1)
#        c = (predicted == labels).squeeze()
#        for i in range(4):
#            label = labels[i]
#            class_correct[label] += c[i].item()
#            class_total[label] += 1
#
#
#for i in range(10):
#    print('Accuracy of %5s : %2d %%' % (
#        classes[i], 100 * class_correct[i] / class_total[i]))
#
#for epoch in range(10):  # loop over the dataset multiple times
#
#    running_loss = 0.0
#    for i, data in enumerate(trainloader, 0):
#        # get the inputs
#        inputs, labels = data
#        inputs, labels = inputs.to(device), labels.to(device)
#        # zero the parameter gradients
#        optimizer.zero_grad()
#
#        # forward + backward + optimize
#        outputs = model(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#
#        # print statistics
#        running_loss += loss.item()
#        if i % 2000 == 1999:    # print every 2000 mini-batches
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 2000))
#            running_loss = 0.0
#
#print('Finished Training')
#
#for epoch in range(10):  # loop over the dataset multiple times
#
#    running_loss = 0.0
#    for i, data in enumerate(trainloader, 0):
#        # get the inputs
#        inputs, labels = data
#        inputs, labels = inputs.to(device), labels.to(device)
#        # zero the parameter gradients
#        optimizer.zero_grad()
#
#        # forward + backward + optimize
#        outputs = model(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#
#        # print statistics
#        running_loss += loss.item()
#        if i % 2000 == 1999:    # print every 2000 mini-batches
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 2000))
#            running_loss = 0.0
#
#print('Finished Training')
