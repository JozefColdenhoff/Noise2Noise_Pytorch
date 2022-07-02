import torch
from Proj_338618_335281.Miniproject_2.model import Sequential, SGD, ReLU, Conv2d, TransposeConv2d, Sigmoid, Adam, MSE
from tqdm import tqdm
import time
import torch.nn as nn

def main():
    time_our_implementation()
    time_pytorch()

def time_our_implementation():
    print("timing our implementation")
    noisy_imgs_1, noisy_imgs_2 = torch.load('/home/pc/PycharmProjects/pythonProject/data/train_data.pkl')

    norm_input = (noisy_imgs_1 / 255).split(128)
    norm_target = (noisy_imgs_2 / 255).split(128)
    labels = ['SGD', 'SGD momentum', 'Adam']
    torch.set_grad_enabled(False)
    for i in range(3):
        s = Sequential(Conv2d(3, 16, 4, stride=2, padding=1),
                       ReLU(),
                       Conv2d(16, 32, 4, stride=2),
                       ReLU(),
                       TransposeConv2d(32, 16, 4, stride=2),
                       ReLU(),
                       TransposeConv2d(16, 3, 4, stride=2, padding=1),
                       Sigmoid())

        s.cuda()
        if i == 0:
            opt = SGD(s, lr=0.5, device='cuda')
        if i == 1:
            opt = SGD(s, lr=0.5, momentum=0.9, device='cuda')
        else:
            opt = Adam(s, lr=1e-3, device='cuda')

        l2_loss = MSE()

        times = []
        for _ in tqdm(range(10), total=10):
            start = time.time()
            for inp, target in zip(norm_input, norm_target):
                inp = inp.to('cuda')
                target = target.to('cuda')
                opt.zero_grad()
                output = s(inp)
                l = l2_loss(target, output)
                l.creator.backward()
                l2_loss.clear_saved_tensors()
                opt.step()
            end = time.time()
            times.append(end-start)
        print(f"Time for {labels[i]} optimizer is {torch.Tensor(times).mean()} with standard deviation {torch.Tensor(times).std()}")

def time_pytorch():
    print("timing pytorch")
    noisy_imgs_1, noisy_imgs_2 = torch.load('/home/pc/PycharmProjects/pythonProject/data/train_data.pkl')

    norm_input = (noisy_imgs_1 / 255).split(128)
    norm_target = (noisy_imgs_2 / 255).split(128)
    labels = ['SGD', 'SGD momentum', 'Adam']
    torch.set_grad_enabled(True)
    for i in range(3):
        s = nn.Sequential(nn.Conv2d(3, 16, 4, stride=2, padding=1),
                       nn.ReLU(),
                       nn.Conv2d(16, 32, 4, stride=2),
                       nn.ReLU(),
                       nn.ConvTranspose2d(32, 16, 4, stride=2),
                       nn.ReLU(),
                       nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
                       nn.Sigmoid())

        s.to('cuda')
        if i == 0:
            opt = torch.optim.SGD(s.parameters(), lr=0.5)
        if i == 1:
            opt = torch.optim.SGD(s.parameters(), lr=0.5, momentum=0.9)
        else:
            opt = torch.optim.Adam(s.parameters(), lr=1e-3)

        l2_loss = nn.MSELoss()

        times = []
        for _ in tqdm(range(10), total=10):
            start = time.time()
            for inp, target in zip(norm_input, norm_target):
                inp = inp.to('cuda')
                target = target.to('cuda')
                opt.zero_grad()
                output = s(inp)
                l = l2_loss(target, output)
                l.backward()
                opt.step()

            end = time.time()
            times.append(end-start)
        print(f"Time for {labels[i]} optimizer is {torch.Tensor(times).mean()} with standard deviation {torch.Tensor(times).std()}")

if __name__ == '__main__':
    main()