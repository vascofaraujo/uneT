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
        level = self.levels[1] #t1ce
        image = nib.load(self.dataset_folder + self.dataset_list[folder_index] + '/' + self.dataset_list[folder_index] + '_' + level + '.nii.gz').get_fdata()
        seg = nib.load(self.dataset_folder + self.dataset_list[folder_index] + '/' + self.dataset_list[folder_index] + '_seg' + '.nii.gz').get_fdata()

        image = image[:,:,image.shape[2]//2]
        image = (image * 255) / np.max(image)

        seg = seg[:,:,seg.shape[2]//2]

        image_pad = np.zeros((256, 256))
        seg_pad = np.zeros((256, 256))

        image_pad[7:247, 7:247] = image
        seg_pad[7:247, 7:247] = seg
        # print(image_pad.shape, seg_pad.shape)
        n = random.randint(0, 127)

        image_pad = image_pad[n:n+128, n:n+128]
        seg_pad = seg_pad[n:n+128, n:n+128]

        image = torch.from_numpy(image_pad)
        seg = torch.from_numpy(seg_pad)

        return image[None,:,:], seg[None,:,:]


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
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

class PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, n_patches, size_patches_h, size_patches_w = x.shape

        x = torch.reshape(x, (batch_size, n_patches, size_patches_h*size_patches_w))
        x = x.transpose(1, 2)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, attention_layers):
        super().__init__()

        in_layers = attention_layers**2
        out_layers = attention_layers**2

        self.query = nn.Linear(in_layers, out_layers)
        self.key = nn.Linear(in_layers, out_layers)
        self.value = nn.Linear(in_layers, out_layers)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention = key @ query.transpose(1, 2)

        output = self.softmax(attention @ value)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, config, attention_layers):
        super().__init__()
        self.position_embedding = PositionalEmbedding()
        self.attention = AttentionBlock(attention_layers)

    def forward(self,x):
        print(x.shape)
        x = self.position_embedding(x)
        print(x.shape)
        x = self.attention(x)

        x = x.transpose(1, 2)

        batch_size, n_patches, size_patches = x.shape
        sqrt_size_patches = int(np.sqrt(size_patches))
        x = torch.reshape(x, (batch_size, n_patches, sqrt_size_patches, sqrt_size_patches))

        return x

class UneT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # encoder_channels: 1, 16, 32, 64, 128, etc
        encoder_channels = [1]
        for i in range(config['n_encoders']):
            encoder_channels.append(config['encoder_channels']*(2**i))

        self.encoder_blocks = nn.ModuleList([EncoderBlock(encoder_channels[i], encoder_channels[i+1], config['encoder_kernel_size'], config['encoder_padding']) for i in range(config['n_encoders'])])

        decoder_kernels = config['decoder_kernel_size']
        for i in range(len(decoder_kernels) - config['n_encoders']):
            decoder_kernels.pop(0)

        self.upsampler_blocks = nn.ModuleList([DecoderBlock(encoder_channels[i+2], encoder_channels[i+1], decoder_kernels[idx], config['decoder_padding']) for idx, i in enumerate(reversed(range(config['n_encoders']-1)))])

        print(encoder_channels)
        self.transformer = TransformerBlock(config, encoder_channels[-1])

        encoder_channels[0] = 4
        self.decoder_blocks = nn.ModuleList([EncoderBlock(encoder_channels[i+1], encoder_channels[i], config['encoder_kernel_size'], config['encoder_padding']) for i in reversed(range(config['n_encoders']))])


        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c = []
        for i, encoder in enumerate(self.encoder_blocks):
            x = encoder(x)
            if i == len(self.encoder_blocks)-1:
                continue
            c.append(x)
            x = self.pool(x)

        #x = self.transformer(x)

        for upsampler, decoder in zip(self.upsampler_blocks, self.decoder_blocks):
            x = upsampler(x)
            if not c:
                continue
            last_c = c.pop(-1)
            x = torch.cat((x, last_c), 1)
            x = decoder(x)

        x = self.decoder_blocks[-1](x)

        return torch.softmax(x, dim=1)

def one_hot_encodding(img, ncols=4):
    out = torch.zeros(img.shape[0], ncols, img.shape[2], img.shape[3])

    out[:,0,:,:] = torch.where(img[:,0,:,:] == 1.0, 1.0, 0.0)
    out[:,1,:,:] = torch.where(img[:,0,:,:] == 2.0, 1.0, 0.0)
    out[:,2,:,:] = torch.where(img[:,0,:,:] == 3.0, 1.0, 0.0)
    out[:,3,:,:] = torch.where(img[:,0,:,:] == 4.0, 1.0, 0.0)

    return out

def show_images(output, label):
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(2, 4, 1)
    plt.imshow(output[0,0,:,:])
    fig.add_subplot(2, 4, 2)
    plt.imshow(output[0,1,:,:])
    fig.add_subplot(2, 4, 3)
    plt.imshow(output[0,2,:,:])
    fig.add_subplot(2, 4, 4)
    plt.imshow(output[0,3,:,:])
    fig.add_subplot(2, 4, 5)
    plt.imshow(label[0,0,:,:])
    fig.add_subplot(2, 4, 6)
    plt.imshow(label[0,1,:,:])
    fig.add_subplot(2, 4, 7)
    plt.imshow(label[0,2,:,:])
    fig.add_subplot(2, 4, 8)
    plt.imshow(label[0,3,:,:])
    plt.show()

def train_model(model, train_dataloader, config):
    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    epochs = config['num_epochs']

    best_loss = 0.2

    epoch_loss = []
    for epoch in tqdm(range(epochs)):
        train_loss = []
        for img, seg in train_dataloader:
            optimizer.zero_grad()

            label = one_hot_encodding(seg)

            output = model(img.float())

            loss = criterion(output.float(), label)
            if loss < 0.2:
                show_images(output.detach().numpy(), label)

            loss.backward()
            optimizer.step()

            print(loss.item())

            train_loss.append(loss.item())
        curr_epoch_loss = np.mean(train_loss)
        print(f"Epoch loss: {curr_epoch_loss}")
        if curr_epoch_loss < best_loss:
            torch.save(model, "uneT.pth")
            best_loss = curr_epoch_loss

if __name__ == '__main__':

    with open("config.json") as config_file:
        config = json.load(config_file)

    train_dataset = BraTSDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = UneT(config)

    _ = train_model(model, train_dataloader, config)
