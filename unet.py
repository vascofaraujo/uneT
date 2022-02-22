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
        all_img = []
        for level in self.levels:
            img = nib.load(self.dataset_folder + self.dataset_list[folder_index] + '/' + self.dataset_list[folder_index] + '_' + level + '.nii.gz').get_fdata()

            # normalize images to be between 0 and 255
            img = (img * 255) / np.max(img)
            all_img.append(img)

        seg = nib.load(self.dataset_folder + self.dataset_list[folder_index] + '/' + self.dataset_list[folder_index] + '_seg' + '.nii.gz').get_fdata()

        img = np.array(all_img)

        n_levels, width, height, n_images = img.shape

        s = self.config['brain_3d_size']
        n = random.randint(50, 170-s) # for 128 should be (0, width-s)
        m = random.randint(50, 80) # for 128 should be (9, n_images-s)
        img = img[:,n:n+s, n:n+s, m:m+s]
        seg = seg[n:n+s, n:n+s, m:m+s]

        img = np.moveaxis(img, 3, 1)
        seg =  np.moveaxis(seg, 2, 0)

        img = torch.from_numpy(img)
        seg = torch.from_numpy(seg)

        # 4, n_images, height, width
        return img[:, :, :, :], seg[None, :, :, :]

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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)

        batch_size, n_feats, depth, height, width = x.shape
        x = torch.reshape(x, (batch_size, n_feats, depth*height*width))

        position_embedding = nn.Parameter(torch.zeros(batch_size, n_feats, depth*height*width))
        x = x + position_embedding

        return x.transpose(1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, config, attention_layers):
        super().__init__()
        self.config = config
        self.d = attention_layers*4

        self.position_embedding = PositionalEmbedding(attention_layers, attention_layers*4)
        self.layer_norm = nn.LayerNorm(self.d)
        self.query = nn.Linear(self.d, self.d)
        self.key = nn.Linear(self.d, self.d)
        self.value = nn.Linear(self.d, self.d)
        self.softmax = nn.Softmax(dim=2)
        self.ffn = nn.Linear(self.d, self.d)
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(self.d, attention_layers, kernel_size=3, padding=1)

    def _split_heads(self, x):
        shape = x.shape
        return x.view(shape[0], shape[1], self.config['transformer_n_heads'], shape[2]//self.config['transformer_n_heads']).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.config['transformer_n_heads'])

    def forward(self,x):
        # for final reshape
        batch_size, n_feats, w, h, d = x.shape

        x = self.position_embedding(x)
        x = self.layer_norm(x)

        query, key, value = self.query(x), self.key(x), self.value(x)

        query, key, value = self._split_heads(query), self._split_heads(key), self._split_heads(value)

        attention = self.softmax((query @ key.transpose(2, 3)) / np.sqrt(query.shape[3])) @ value

        attention = self._merge_heads(attention)

        xt = attention + x

        x = self.layer_norm(xt)

        x = self.relu(self.ffn(x)) + xt

        x = x.transpose(1, 2)
        x = torch.reshape(x, (batch_size, self.d, w, h, d))

        x = self.conv(x)
        return x

class UneT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # encoder_channels: 4, 16, 32, 64, etc
        encoder_channels = [4]
        for i in range(config['n_encoders']):
            encoder_channels.append(config['encoder_channels']*(2**i))

        self.encoder_blocks = nn.ModuleList([EncoderBlock(encoder_channels[i], encoder_channels[i+1], config['encoder_kernel_size'], config['encoder_padding']) for i in range(config['n_encoders'])])

        decoder_kernels = config['decoder_kernel_size']
        for i in range(len(decoder_kernels) - config['n_encoders']):
            decoder_kernels.pop(0)

        self.upsampler_blocks = nn.ModuleList([DecoderBlock(encoder_channels[i+2], encoder_channels[i+1], decoder_kernels[idx], config['decoder_padding']) for idx, i in enumerate(reversed(range(config['n_encoders']-1)))])

        mlp_layers = encoder_channels[-1]
        self.transformer = TransformerBlock(config, mlp_layers)

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
