from torch.utils.data import DataLoader
from torchvision import transforms

import DataSet
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

root_train_txt = "/home/freeaccess/Desktop/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
root_test_txt = "/home/freeaccess/Desktop/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
root_segm = "/home/freeaccess/Desktop/VOCdevkit/VOC2012/SegmentationClass"
root_images = "/home/freeaccess/Desktop/VOCdevkit/VOC2012/JPEGImages"

transform_input_image = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])

transform_target_image = transforms.Compose([transforms.Resize((256, 256)),
                                             #transforms.ToTensor(),
                                             ])

# transform_target_image = transforms.Compose([transforms.Resize((256, 256))])


train_set = DataSet.ImageSegmentationDataset(root_train_txt, root_segm, root_images,
                                             transform_input_image, transform_target_image)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

test_set = DataSet.ImageSegmentationDataset(root_test_txt, root_segm, root_images,
                                            transform_input_image, transform_target_image)

test_loader = DataLoader(test_set, batch_size=8, shuffle=True)

model = torch.load('/home/freeaccess/PycharmProjects/VOCPascal/Last folder/new60').cuda() #BEST VAL_LOSS
#model = torch.load('/home/freeaccess/PycharmProjects/VOCPascal/Final weights/Final').cpu() #BEST Train_Loss
#model = torch.load('/home/freeaccess/PycharmProjects/VOCPascal/Last folder/val_loss_try_second0').cpu()
model.eval()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print(params)

criterion = nn.CrossEntropyLoss()

valid_loss = 0
correct = 0
total = 0

#for batch_idx, (inputs, targets) in enumerate(train_loader):
for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs, targets = inputs.cuda(), targets.cuda()

    outputs = model(inputs)

    loss = criterion(outputs, targets.squeeze(dim=1).long())

    valid_loss += loss.item()

    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data.long()).cpu().sum()

    f = plt.figure()
    ax1 = f.add_subplot(2, 2, 1)
    ax1.title.set_text('Input Image')
    plt.imshow(inputs[0][0].cpu())
    ax2 = f.add_subplot(2, 2, 2)
    ax2.title.set_text('Target Image')
    plt.imshow(targets[0].data.cpu())
    ax3 = f.add_subplot(2, 2, 3)
    ax3.title.set_text('Predicted Image')
    plt.imshow(predicted[0].cpu())
    plt.show()

print('Validation Loss: %.3f | Validation Acc: %.3f%% (%d/%d)'
      % (valid_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

model.train()