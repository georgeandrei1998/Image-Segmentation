from PIL import Image
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF


class ImageSegmentationDataset(Dataset):
    def __init__(self, root_dir_names, root_dir_segm, root_dir_jpg, transform_input_image, transform_target_image):
        data = pd.read_csv(root_dir_names, sep="\n", engine='python', header=None).values.flatten()
        self.image_names = data
        self.root_dir_names = root_dir_names
        self.root_dir_segm = root_dir_segm
        self.root_dir_jpg = root_dir_jpg
        self.transform_input_image = transform_input_image
        self.transform_target_image = transform_target_image

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        input_image = Image.open(self.root_dir_jpg + '/' + self.image_names[idx] + '.jpg')
        segm_image = Image.open(self.root_dir_segm + '/' + self.image_names[idx] + '.png')

        if (random.random() > 0.5):
            input_image = TF.hflip(input_image)
            segm_image = TF.hflip(segm_image)

        # if (random.random() > 0.5):
        #     input_image = TF.vflip(input_image)
        #     segm_image = TF.vflip(segm_image)

        input_image = self.transform_input_image(input_image)

        segm_image = np.array(self.transform_target_image(segm_image))
        segm_image[segm_image == 255] = 0

        return input_image, segm_image
