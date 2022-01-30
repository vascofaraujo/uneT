import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import json

class BraTSDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_folder = './brats2021/'
        self.dataset_list = os.listdir('./brats2021/')
        self.levels = ['t1', 'flair', 't2', 't1ce']


    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, folder_index):
        level = self.levels[1] #t1ce
        image = nib.load(self.dataset_folder + self.dataset_list[folder_index] + '/' + self.dataset_list[folder_index] + '_' + level + '.nii.gz').get_fdata()
        seg = nib.load(self.dataset_folder + self.dataset_list[folder_index] + '/' + self.dataset_list[folder_index] + '_seg' + '.nii.gz').get_fdata()

        image = (image * 255) / np.max(image)

        width, height, n_images = image.shape

        s = self.config['brain_3d_size']
        n = random.randint(s, 200-s) # for 128 should be (0, width-s)
        m = random.randint(50, 130-s) # for 128 should be (9, n_images-s)
        image = image[n:n+s, n:n+s, m:m+s]
        seg = seg[n:n+s, n:n+s, m:m+s]

        image = np.moveaxis(image, 2, 0)
        seg =  np.moveaxis(seg, 2, 0)

        image = torch.from_numpy(image)
        seg = torch.from_numpy(seg)

        # 1, n_images, height, width
        return image[None, :, :, :], seg[None, :, :, :]

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.relu(self.conv2(x))

        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.transposed_conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.transposed_conv(x))
        x = self.relu(self.transposed_conv2(x))

        return x

class PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, n_classes, depth, height, width = x.shape

        x = torch.reshape(x, (batch_size, n_classes, depth*height*width))
        x = x.transpose(1, 2)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, attention_layers):
        super().__init__()

        self.attention_layers = attention_layers
        in_layers = attention_layers
        out_layers = attention_layers

        self.query = nn.Linear(in_layers, out_layers)
        self.key = nn.Linear(in_layers, out_layers)
        self.value = nn.Linear(in_layers, out_layers)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        n_batches, w, h = x.shape
        query = self.query(torch.reshape(x, (n_batches, w*h)))
        key = self.key(torch.reshape(x, (n_batches, w*h)))
        value = self.value(torch.reshape(x, (n_batches, w*h)))

        query, key, value = torch.reshape(query, (n_batches, w, h)), torch.reshape(key, (n_batches, w, h)), torch.reshape(value, (n_batches, w, h))

        attention = self.softmax((query @ key.transpose(1, 2)) / np.sqrt(self.attention_layers)) @ value

        return x

class TransformerBlock(nn.Module):
    def __init__(self, config, attention_layers):
        super().__init__()
        self.position_embedding = PositionalEmbedding()
        self.attention = AttentionBlock(attention_layers)

    def forward(self,x):
        x = self.position_embedding(x)
        x = self.attention(x)

        x = x.transpose(1, 2)

        batch_size, n_patches, size_patches = x.shape
        new_size_patches = int(np.cbrt(size_patches))
        x = torch.reshape(x, (batch_size, n_patches, new_size_patches, new_size_patches, new_size_patches))

        return x

class UneT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # encoder_channels: 1, 16, 32, 64, etc
        encoder_channels = [1]
        for i in range(config['n_encoders']):
            encoder_channels.append(config['encoder_channels']*(2**i))

        self.encoder_blocks = nn.ModuleList([EncoderBlock(encoder_channels[i], encoder_channels[i+1], config['encoder_kernel_size'], config['encoder_padding']) for i in range(config['n_encoders'])])

        decoder_kernels = config['decoder_kernel_size']
        for i in range(len(decoder_kernels) - config['n_encoders']):
            decoder_kernels.pop(0)

        self.upsampler_blocks = nn.ModuleList([DecoderBlock(encoder_channels[i+2], encoder_channels[i+1], decoder_kernels[idx], config['decoder_padding']) for idx, i in enumerate(reversed(range(config['n_encoders']-1)))])

        self.transformer = TransformerBlock(config, encoder_channels[-1]*encoder_channels[-2])

        encoder_channels[0] = config['num_classes']
        self.decoder_blocks = nn.ModuleList([EncoderBlock(encoder_channels[i+1], encoder_channels[i], config['encoder_kernel_size'], config['encoder_padding']) for i in reversed(range(config['n_encoders']))])

        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        c = []
        for i, encoder in enumerate(self.encoder_blocks):
            x = encoder(x)
            if i == len(self.encoder_blocks)-1:
                continue
            c.append(x)
            x = self.pool(x)

        x = self.transformer(x)

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
    out = torch.zeros(img.shape[0], ncols, img.shape[2], img.shape[3], img.shape[4])

    out[:,0,:,:,:] = torch.where(img[:,0,:,:,:] == 1.0, 1.0, 0.0)
    out[:,1,:,:,:] = torch.where(img[:,0,:,:,:] == 2.0, 1.0, 0.0)
    out[:,2,:,:,:] = torch.where(img[:,0,:,:,:] == 3.0, 1.0, 0.0)
    out[:,3,:,:,:] = torch.where(img[:,0,:,:,:] == 4.0, 1.0, 0.0)

    return out

def show_images(output, label):
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(2, 4, 1)
    plt.imshow(output[0,0,16, :,:])
    fig.add_subplot(2, 4, 2)
    plt.imshow(output[0,1,16,:,:])
    fig.add_subplot(2, 4, 3)
    plt.imshow(output[0,2,16,:,:])
    fig.add_subplot(2, 4, 4)
    plt.imshow(output[0,3,16,:,:])
    fig.add_subplot(2, 4, 5)
    plt.imshow(label[0,0,16,:,:])
    fig.add_subplot(2, 4, 6)
    plt.imshow(label[0,1,16,:,:])
    fig.add_subplot(2, 4, 7)
    plt.imshow(label[0,2,16,:,:])
    fig.add_subplot(2, 4, 8)
    plt.imshow(label[0,3,16,:,:])
    plt.show()

class DiceLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.epsilon = config['loss_epsilon']
        self.num_classes = config['num_classes']

    def forward(self, output, target):

        batch_size, n_classes, depth, height, width = output.shape
        a = []
        for i in range(4):
            G = output[:,i,:,:,:].view(batch_size, depth*height*width)
            Y = target[:,i,:,:,:].view(batch_size, depth*height*width)

            num = torch.sum(torch.mul(G, Y), dim=1) + self.epsilon
            den = torch.sum(G.pow(2), dim=1) + torch.sum(Y.pow(2), dim=1) + self.epsilon
            a.append(num/den)
        loss = 1 - (2/self.num_classes)*(sum(a))

        return loss.mean()


def train_model(model, train_dataloader, config):
    criterion = DiceLoss(config)

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

            loss = criterion(output , label)
            print(loss)

            # show_images(output.detach().numpy(), label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        curr_epoch_loss = np.mean(train_loss)
        print(f"Epoch loss: {curr_epoch_loss}")
        if curr_epoch_loss < best_loss:
            torch.save(model, "uneT.pth")
            best_loss = curr_epoch_loss

if __name__ == '__main__':

    with open("config.json") as config_file:
        config = json.load(config_file)

    train_dataset = BraTSDataset(config)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = UneT(config)

    _ = train_model(model, train_dataloader, config)
