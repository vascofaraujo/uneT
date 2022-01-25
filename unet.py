import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import cv2 as cv
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import json

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

        image = image[:,:,image.shape[2]//2]
        image = (image * 255) / np.max(image)

        seg = seg[:,:,seg.shape[2]//2]

        image_pad = np.zeros((256, 256))
        seg_pad = np.zeros((256, 256))

        image_pad[7:247, 7:247] = image
        seg_pad[7:247, 7:247] = seg

        n = random.randint(0, 191)

        image_pad = image_pad[n:n+64, n:n+64]
        seg_pad = seg_pad[n:n+64, n:n+64]

        image = torch.from_numpy(image_pad)
        seg = torch.from_numpy(seg_pad)

        return image[None,:,:], seg[None,:,:]


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.relu(self.conv2(x))

        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.transposed_conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.transposed_conv(x))
        x = self.relu(self.transposed_conv2(x))

        return x


# class PositionalEmbedding(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.batch_size = config['batch_size']
#         self.n_patches = config['n_patches']
#         self.size_patches = config['size_patches']
#
#
#     def forward(self, x):
#         # Receives 16x256x4x4
#         x = torch.reshape(x, (self.batch_size, self.n_patches, self.size_patches)) # 16x256x16
#         x = x.transpose(1, 2) # 16x16x256
#
#         return x
#
#
# class AttentionBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#
#         in_layers = config['transformer_in_layers']
#         out_layers = config['transformer_out_layers']
#
#         self.query = nn.Linear(in_layers, out_layers)
#         self.key = nn.Linear(in_layers, out_layers)
#         self.value = nn.Linear(in_layers, out_layers)
#         self.softmax = nn.Softmax(dim=2)
#
#
#
#     def forward(self, x):
#         batch_size, size_patches, n_patches = x.shape
#
#         query = self.query(x)
#         key = self.key(x)
#         value = self.value(x)#16x16x256
#
#
#         attention = key @ query.transpose(1, 2)
#         # 16x16x16
#
#         output = self.softmax(attention @ value)
#
#         return x
#
# class TransformerBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.batch_size = config['batch_size']
#         self.transformer_out_layers = config['transformer_out_layers']
#
#         self.position_embedding = PositionalEmbedding(config)
#         self.attention = AttentionBlock(config)
#
#     def forward(self,x):
#         x = self.position_embedding(x)
#         x = self.attention(x)
#
#         x = x.transpose(1, 2)
#         x = torch.reshape(x, (self.batch_size, self.transformer_out_layers, 4, 4))
#         return x

class UneT(nn.Module):
    def __init__(self, config):
        super().__init__()

        num_channels = config['encoder_channels'] #better name for this variable?
        kernel_size = config['kernel_size']


        self.encoder1 = EncoderBlock(in_channels=1, out_channels=num_channels)
        self.encoder2 = EncoderBlock(in_channels=num_channels, out_channels=num_channels*2)
        self.encoder3 = EncoderBlock(in_channels=num_channels*2, out_channels=num_channels*4)
        self.encoder4 = EncoderBlock(in_channels=num_channels*4, out_channels=num_channels*8)
        self.encoder5 = EncoderBlock(in_channels=num_channels*8, out_channels=num_channels*16)

        # self.transformer = TransformerBlock(config)

        self.encoder6 = EncoderBlock(in_channels=num_channels*16, out_channels=num_channels*8)
        self.encoder7 = EncoderBlock(in_channels=num_channels*8, out_channels=num_channels*4)
        self.encoder8 = EncoderBlock(in_channels=num_channels*4, out_channels=num_channels*2)
        self.encoder9 = EncoderBlock(in_channels=num_channels*2, out_channels=num_channels)
        self.encoder10 = EncoderBlock(in_channels=num_channels, out_channels=4)

        print(kernel_size)
        self.decoder1 = DecoderBlock(in_channels=num_channels*16, out_channels=num_channels*8, kernel_size=kernel_size[0], padding=0)
        self.decoder2 = DecoderBlock(in_channels=num_channels*8, out_channels=num_channels*4, kernel_size=kernel_size[1], padding=0)
        self.decoder3 = DecoderBlock(in_channels=num_channels*4, out_channels=num_channels*2, kernel_size=kernel_size[2], padding=0)
        self.decoder4 = DecoderBlock(in_channels=num_channels*2, out_channels=num_channels, kernel_size=kernel_size[3], padding=0)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        c1 = self.encoder1(x)
        p1 = self.pool(c1)

        c2 = self.encoder2(p1)
        p2 = self.pool(c2)

        c3 = self.encoder3(p2)
        # p3 = self.pool(c3)
        #
        # c4 = self.encoder4(p3)
        # p4 = self.pool(c4)
        #
        # c5 = self.encoder5(p4)

        # x = self.transformer(c5)

        # u6 = self.decoder1(c5)
        #
        # p6 = torch.cat((u6, c4), 1)

        # c6 = self.encoder6(p6)
        # u7 = self.decoder2(c6)
        # p7 = torch.cat((u7, c3), 1)

        # c7 = self.encoder7(c3)
        print(f"Before upsampler : {c3.shape}")
        u8 = self.decoder3(c3)
        print(f"After upsample: {u8.shape}")
        p8 = torch.cat((u8, c2), 1)

        c8 = self.encoder8(p8)
        print(f"Before upsampler : {c8.shape}")
        u9 = self.decoder4(c8)
        print(f"After upsample: {u9.shape}")
        p9 = torch.cat((u9, c1), 1)

        c9 = self.encoder9(p9)
        y = self.encoder10(c9)
        print(y.shape)

        return torch.softmax(y, dim=1)

def one_hot_encodding(img, ncols=4):
    out = torch.zeros(img.shape[0], ncols, img.shape[2], img.shape[3])

    out[:,0,:,:] = torch.where(img[:,0,:,:] == 1.0, 1.0, 0.0)
    out[:,1,:,:] = torch.where(img[:,0,:,:] == 2.0, 1.0, 0.0)
    out[:,2,:,:] = torch.where(img[:,0,:,:] == 3.0, 1.0, 0.0)
    out[:,3,:,:] = torch.where(img[:,0,:,:] == 4.0, 1.0, 0.0)

    return out

def train_model(model, train_dataloader, config):
    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    epochs = config['num_epochs']

    epoch_loss = []
    for epoch in tqdm(range(epochs)):
        train_loss = []

        for img, seg in train_dataloader:
            optimizer.zero_grad()

            label = one_hot_encodding(seg)

            output = model(img.float())

            loss = criterion(output.float(), label)

            output = output.detach().numpy()

            loss.backward()
            optimizer.step()

            #print(loss.item())

            train_loss.append(loss.item())
        curr_epoch_loss = np.mean(train_loss)
        print(f"Epoch loss: {curr_epoch_loss}")



if __name__ == '__main__':

    with open("config.json") as config_file:
        config = json.load(config_file)

    train_dataset = BraTSDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = UneT(config)

    _ = train_model(model, train_dataloader, config)



    # fig = plt.figure(figsize=(8, 8))
    # fig.add_subplot(2, 4, 1)
    # plt.imshow(output[0,0,:,:])
    # fig.add_subplot(2, 4, 2)
    # plt.imshow(output[0,1,:,:])
    # fig.add_subplot(2, 4, 3)
    # plt.imshow(output[0,2,:,:])
    # fig.add_subplot(2, 4, 4)
    # plt.imshow(output[0,3,:,:])
    # fig.add_subplot(2, 4, 5)
    # plt.imshow(label[0,0,:,:])
    # fig.add_subplot(2, 4, 6)
    # plt.imshow(label[0,1,:,:])
    # fig.add_subplot(2, 4, 7)
    # plt.imshow(label[0,2,:,:])
    # fig.add_subplot(2, 4, 8)
    # plt.imshow(label[0,3,:,:])
    # plt.show()
