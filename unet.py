import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import cv2 as cv
import matplotlib.pyplot as plt

class BraTSDataset(Dataset):
    def __init__(self):
        self.dataset_folder = './brats2021/'
        self.dataset_list = os.listdir('./brats2021/')
        self.levels = ['t1', 'flair', 't2', 't1ce']


    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, folder_index):
        level = self.levels[3] #t1ce
        image = nib.load(self.dataset_folder + self.dataset_list[folder_index] + '/' + self.dataset_list[folder_index] + '_' + level + '.nii.gz').get_fdata()
        seg = nib.load(self.dataset_folder + self.dataset_list[folder_index] + '/' + self.dataset_list[folder_index] + '_seg' + '.nii.gz').get_fdata()
        image = cv.resize(image[:,:,image.shape[2]//2], (128,128), interpolation=cv.INTER_LINEAR)
        image = (image * 255) / np.max(image)

        seg = cv.resize(seg[:,:,seg.shape[2]//2], (128,128), interpolation=cv.INTER_LINEAR)

        image = torch.from_numpy(image)
        seg = torch.from_numpy(seg)

        return image[None,:,:], seg[None,:,:]


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.transposed_conv(x)
        x = self.relu(x)

        return x

class UneT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = EncoderBlock(in_channels=1, out_channels=16)
        self.encoder2 = EncoderBlock(in_channels=16, out_channels=32)
        self.encoder3 = EncoderBlock(in_channels=32, out_channels=64)
        self.encoder4 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder5 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder6 = EncoderBlock(in_channels=256, out_channels=128)
        self.encoder7 = EncoderBlock(in_channels=128, out_channels=64)
        self.encoder8 = EncoderBlock(in_channels=64, out_channels=32)
        self.encoder9 = EncoderBlock(in_channels=32, out_channels=16)
        self.encoder10 = EncoderBlock(in_channels=16, out_channels=4)

        self.decoder1 = DecoderBlock(in_channels=256, out_channels=128, kernel_size=9, padding=0)
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64, kernel_size=17, padding=0)
        self.decoder3 = DecoderBlock(in_channels=64, out_channels=32, kernel_size=33, padding=0)
        self.decoder4 = DecoderBlock(in_channels=32, out_channels=16, kernel_size=65, padding=0)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c1 = self.encoder1(x)
        p1 = self.pool(c1)

        c2 = self.encoder2(p1)
        p2 = self.pool(c2)

        c3 = self.encoder3(p2)
        p3 = self.pool(c3)

        c4 = self.encoder4(p3)
        p4 = self.pool(c4)

        c5 = self.encoder5(p4)

        u6 = self.decoder1(c5)
        p6 = torch.cat((u6, c4), 1)

        c6 = self.encoder6(p6)
        u7 = self.decoder2(c6)
        p7 = torch.cat((u7, c3), 1)

        c7 = self.encoder7(p7)
        u8 = self.decoder3(c7)
        p8 = torch.cat((u8, c2), 1)

        c8 = self.encoder8(p8)
        u9 = self.decoder4(c8)
        p9 = torch.cat((u9, c1), 1)

        c9 = self.encoder9(p9)
        y = self.encoder10(c9)

        print(y.shape)
        return torch.softmax(y, dim=1)

def one_hot_encodding(img, ncols=4):
    print(img.shape)
    out = torch.zeros(img.shape[0], ncols, img.shape[2], img.shape[3])
    # out[torch.arange(img.size, im] = 1
    return out

def train_model(model, train_dataloader):
    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for img, seg in train_dataloader:
        #break
        #print(img)
        optimizer.zero_grad()
        label = one_hot_encodding(seg)
        output = model(img.float())
        plt.imshow(output[0,0,:,:].detach().numpy())
        loss = criterion(output.float(), label)
        print(loss.item())
        loss.backward()
        optimizer.step()


    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img[0,0,:,:].T, cmap='Greys_r')
    fig.add_subplot(1, 2, 2)
    plt.imshow(seg[0,0,:,:].T, cmap='Greys_r')
    plt.show()

if __name__ == '__main__':

    train_dataset = BraTSDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = UneT()

    _ = train_model(model, train_dataloader)



    # seg = seg.detach().numpy()
    # target = target.detach().numpy()
    #
    # fig = plt.figure(figsize=(8, 8))
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(seg[0,0,:,:].T, cmap='Greys_r')
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(target[0,0,:,:].T, cmap='Greys_r')
    # plt.show()
