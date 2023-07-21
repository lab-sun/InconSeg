# By Yuxiang Sun, Jul. 3, 2021
# Email: sun.yuxiang@outlook.com

import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL

class RGB_Disparity_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=288, input_w=512 ,transform=[]):
        super(RGB_Disparity_dataset, self).__init__()

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.n_data    = len(self.names)

    def read_image(self, name, folder,head):
        file_path = os.path.join(self.data_dir, '%s/%s%s.png' % (folder, head,name))
        image     = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'left','left')
        label = self.read_image(name, 'labels','label')
        depth = self.read_image(name, 'disparity','disparity')
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)))
        image = image.astype('float32')
        image = np.transpose(image, (2,0,1))/255.0
        depth = np.asarray(PIL.Image.fromarray(depth).resize((self.input_w, self.input_h)))
        depth = depth.astype('float32')
        M = depth.max()
        depth = depth/M
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
        label = label.astype('int64')

        return torch.cat((torch.tensor(image), torch.tensor(depth).unsqueeze(0)),dim=0), torch.tensor(label), name

    def __len__(self):
        return self.n_data

