import random
import torch.nn as nn
import torch
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop, Compose, Pad, Grayscale

class Model():
    def __init__(self):
        self.net = ConcatModel(n_layers=5, up_ch=10, kernel_size=2, activation_fn=F.mish)
        self.cuda = False
        try:
            from torch.cuda import is_available as has_cuda
            if has_cuda():
                self.cuda = True
        except:
            self.cuda = False

        if self.cuda:
            self.net.cuda()

        self.batch_size = 16
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 15], gamma=0.5)

    def load_pretrained_model(self):
        from pathlib import Path
        model_path = Path(__file__).parent / "bestmodel.pth"
        if self.cuda:
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        self.net.load_state_dict(state_dict)
        if self.cuda:
            self.net.cuda()


    def train(self, train_input, train_target, num_epochs):
        train_set = NoisyDataSet(train_input, train_target)
        train_loader = DataLoader(train_set, batch_size=self.batch_size)

        self.net.train()
        for _ in range(num_epochs):
            for inputs, targets in train_loader:
                if self.cuda:
                    inputs = inputs.to('cuda')
                    targets = targets.to('cuda')
                self.optimizer.zero_grad()
                output = self.net(inputs)
                l = self.loss(targets, output)
                l.backward()
                self.optimizer.step()
            self.scheduler.step()

    def predict(self, test_input):
        self.net.eval()
        with torch.no_grad():
            if self.cuda:
                out = (self.net(test_input.to('cuda') / 255) * 255)
                return torch.clip(out, 0, 255)
            else:
                return torch.clip(self.net(test_input / 255) * 255, 0, 255)

    def save(self, checkpointpath):
        torch.save(self.net.state_dict(), checkpointpath)

class ConcatModel(nn.Module):
    def  __init__(self, n_layers=5, up_ch=10, kernel_size=2, activation_fn=F.relu):
        super(ConcatModel, self).__init__()
        self.n_layers = n_layers
        self.up_ch = up_ch
        self.activation_fn = activation_fn
        self.mod_list = nn.ModuleList(
            [nn.Conv2d(3, up_ch, kernel_size)]
        )
        if n_layers >= 2:
            self.mod_list += [nn.Conv2d(up_ch * 2 ** (x), up_ch * 2 ** (1 + x), kernel_size) for x in range(n_layers - 1)]
            self.mod_list += [nn.ConvTranspose2d(up_ch * 2 ** (n_layers - x), up_ch * 2 ** (n_layers - 2 - x), kernel_size) for x in range(n_layers-1)]
        self.mod_list += [nn.ConvTranspose2d(2 * up_ch, 3, kernel_size)]

    def forward(self, x):
        activations = []
        for i in range(self.n_layers):
            x = self.activation_fn(self.mod_list[i](x))
            activations.append(x)

        for i in range(self.n_layers - 1):
            x = self.activation_fn(self.mod_list[self.n_layers + i](torch.cat((x, activations[-(i+1)]),dim=1)))
        x = (self.mod_list[2*self.n_layers - 1](torch.cat((x, activations[-self.n_layers]),dim=1))).sigmoid()
        return x


class NoisyDataSet(Dataset):
    def __init__(self, im1, im2, scale=True, test=False, flip=False, crop=False, greyscale=False):
        assert im1.shape == im2.shape
        self.test = test
        self.flip = flip
        self.crop = crop
        self.grey = greyscale
        self.greyscaler = Grayscale(num_output_channels=3)
        self.cropper = Compose([RandomCrop(size=24), Pad((32-24)//2)])
        if scale:
            self.im1 = im1 / 255
            self.im2 = im2 / 255
        else:
            self.im1 = im1
            self.im2 = im2

    def __len__(self):
        return self.im1.shape[0]

    def __getitem__(self, idx):
        if self.test:
            return self.im1[idx], self.im2[idx]
        else:
            im_1, im_2 = self.im1[idx], self.im2[idx]

            if self.flip:
                if random.random() > 0.5:
                    im_1, im_2 = torch.flip(im_1, dims=[1]), torch.flip(im_2, dims=[1])
                if random.random() > 0.5:
                    im_1, im_2 = torch.flip(im_1, dims=[2]), torch.flip(im_2, dims=[2])
            if self.crop:
                if random.random() > 0.5:
                    cropped = self.cropper(torch.cat((im_1, im_2), dim=0))
                    im_1, im_2 = cropped[:3], cropped[3:]
            if self.grey:
                if random.random() > 0.5:
                    im_1, im_2 = self.greyscaler(im_1), self.greyscaler(im_2)

            return im_1, im_2