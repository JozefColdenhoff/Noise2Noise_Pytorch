"""
File to test different optimizer convergence rates
"""

from tqdm import tqdm
import torch
from Proj_338618_335281.Miniproject_2.model import Sequential, SGD, ReLU, Conv2d, TransposeConv2d, Sigmoid, Adam, MSE
import matplotlib.pyplot as plt

def psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()

def main():
    noisy_imgs_1, noisy_imgs_2 = torch.load('/home/pc/PycharmProjects/pythonProject/data/train_data.pkl')
    val_input, val_target = torch.load("/home/pc/PycharmProjects/pythonProject/data/val_data.pkl")
    test_len = 1000
    norm_input = (noisy_imgs_1 / 255).split(16)
    norm_target = (noisy_imgs_2 / 255).split(16)
    norm_val_input, norm_val_target = (val_input / 255).split(100), (val_target /255).split(100)

    optimizer_psnrs = []
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
            opt = Adam(s, lr=1e-2, device='cuda')

        l2_loss = MSE()
        torch.set_grad_enabled(False)
        psnrs = []
        for epoch in tqdm(range(10), total=10):
            for inp, target in zip(norm_input, norm_target):
                inp = inp.to('cuda')
                target = target.to('cuda')
                opt.zero_grad()
                output = s(inp)
                l = l2_loss(target, output)
                l.creator.backward()
                l2_loss.clear_saved_tensors()
                opt.step()

            val_psnr = 0
            for inp, target in zip(norm_val_input, norm_val_target):
                inp = inp.to('cuda')
                val_psnr += psnr(s(inp).cpu(), target) * inp.shape[0]
            psnrs.append((val_psnr / test_len).cpu().item())
        print(val_psnr / test_len)
        optimizer_psnrs.append(psnrs)

    labels = ['SGD', 'SGD momentum', 'Adam']

    for i, lis in enumerate(optimizer_psnrs):
        plt.plot(lis, label=labels[i])

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('validation PSNR')
    plt.savefig("Optimizers_proj2")

if __name__ == '__main__':
    main()