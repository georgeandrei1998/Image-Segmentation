import DataSet
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import Network
import torch
import PIL
import matplotlib.pyplot as plt
import numpy as np
import cv2

torch.backends.cudnn.enable = False

# paths
root_train_txt = "/home/freeaccess/Desktop/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
root_test_txt = "/home/freeaccess/Desktop/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
root_segm = "/home/freeaccess/Desktop/VOCdevkit/VOC2012/SegmentationClass"
root_images = "/home/freeaccess/Desktop/VOCdevkit/VOC2012/JPEGImages"

# transforms
transform_input_image = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.ColorJitter(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])

transform_target_image = transforms.Compose([transforms.Resize((256, 256)),
                                             # transforms.ToTensor()
                                             ])

train_set = DataSet.ImageSegmentationDataset(root_train_txt, root_segm, root_images,
                                             transform_input_image, transform_target_image)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

test_set = DataSet.ImageSegmentationDataset(root_test_txt, root_segm, root_images,
                                            transform_input_image, transform_target_image)

test_loader = DataLoader(test_set, batch_size=8, shuffle=True)

#net = Network.UNet(in_channel=3, out_channel=21).cuda()
net = torch.load('/home/freeaccess/PycharmProjects/VOCPascal/Last folder/val_loss_try_second60')

'''def one_hot(batch_idx, target, class_count):
    y=torch.empty(4,256, 256, 21, dtype=torch.long)
    y.fill_(0)
    for i in range(256):
        for j in range(256):
            y[batch_idx][i][j][target[batch_idx][i][j]] = 1

def lossFunction(output, target):
    return sum(torch.log10_(output)*one_hot(target, 21))'''


def valid_loss_function(net, criterion, test_loader):
    total_loss=0
    for val_batch_idx, (val_inputs, val_targets) in enumerate(test_loader):

        val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()

        net.eval()
        with torch.no_grad():
            val_outputs = net(val_inputs).cuda()
            val_loss = criterion(val_outputs, val_targets.long())
            total_loss += val_loss.item()

    return total_loss/(val_batch_idx+1)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
num_epochs = 100
my_loss_plt = []
my_val_loss = []

for epoch in range(num_epochs):
    net.train()
    train_loss = 0
    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)
        outputs = outputs.cuda()

        loss = criterion(outputs, targets.long())
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data.long()).cpu().sum()

    if epoch % 10 == 0:
        torch.save(net, '/home/freeaccess/PycharmProjects/VOCPascal/Last folder/val_loss_try_second' + str(epoch))

    print('Results after epoch %d' % (epoch + 1))

    print('Training Loss: %.3f | Training Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    my_loss_plt.append(train_loss / (batch_idx + 1))

    if epoch % 10 == 0:
        val_loss = valid_loss_function(net, criterion, test_loader)
        print('Validation loss: %.3f' % val_loss)
        my_val_loss.append(val_loss)

torch.save(net, '/home/freeaccess/PycharmProjects/VOCPascal/Last folder/Val_Final_second')

plt.plot(range(num_epochs), my_loss_plt)
plt.plot(range(0, num_epochs, 10), my_val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
