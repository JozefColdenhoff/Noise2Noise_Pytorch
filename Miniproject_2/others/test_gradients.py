#%%
from tqdm import tqdm
import torch.nn as nn
import torch
from Proj_338618_335281.Miniproject_2.model import Sequential, Conv2d, ReLU, SGD, TransposeConv2d, Sigmoid, Adam, MSE


def main():
    torch.set_default_dtype(torch.float64)
    print("testing SGD with momentum")
    test_SGD()
    print("testing Adam")
    test_Adam()
    print("Passed testing")

def test_SGD():
    s = Sequential(Conv2d(3, 32, 4, stride=2, padding=1),
                   ReLU(),
                   Conv2d(32, 8, 4, stride=2),
                   ReLU(),
                   TransposeConv2d(8, 8, 4, stride=2),
                   ReLU(),
                   TransposeConv2d(8, 3, 4, stride=2, padding=1),
                   Sigmoid())

    s_ = nn.Sequential(nn.Conv2d(3, 32, 4, stride=2, padding=1),
                       nn.ReLU(),
                       nn.Conv2d(32, 8, 4, stride=2),
                       nn.ReLU(),
                       nn.ConvTranspose2d(8, 8, 4, stride=2),
                       nn.ReLU(),
                       nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1),
                       nn.Sigmoid())

    with torch.no_grad():
        for p, p_ in zip(s.param(), s_.parameters()):
            p.copy_(p_.data.clone())

    s_.to('cuda')
    s.cuda()
    opt = SGD(s, lr=0.5, device='cuda', momentum=0.9)
    torch_opt = torch.optim.SGD(s_.parameters(), lr=0.5, momentum=0.9)
    noisy_imgs_1, noisy_imgs_2 = torch.load('/home/pc/PycharmProjects/pythonProject/data/train_data.pkl')

    l2_loss = MSE()
    torch_l2_loss = nn.MSELoss()

    norm_input = (noisy_imgs_1 / 255).split(128)
    norm_target = (noisy_imgs_2 / 255).split(128)

    for inputs, targets in tqdm(zip(norm_input, norm_target), total=len(norm_input)):
        torch.set_grad_enabled(False)
        inp = inputs.to('cuda')
        target = targets.to('cuda')
        opt.zero_grad()
        output = s(inp)
        l = l2_loss(target, output)
        l.creator.backward()
        l2_loss.clear_saved_tensors()
        opt.step()

        torch.set_grad_enabled(True)
        torch_opt.zero_grad()
        output = s_(inp)
        l = torch_l2_loss(target, output)
        l.backward()
        torch_opt.step()

        for p, p_ in zip(s.param(), s_.parameters()):
            assert torch.allclose(p, p_), "Parameters diverged for SGD"

def test_Adam():
    s = Sequential(Conv2d(3, 32, 4, stride=2, padding=1),
                   ReLU(),
                   Conv2d(32, 8, 4, stride=2),
                   ReLU(),
                   TransposeConv2d(8, 8, 4, stride=2),
                   ReLU(),
                   TransposeConv2d(8, 3, 4, stride=2, padding=1),
                   Sigmoid())

    s_ = nn.Sequential(nn.Conv2d(3, 32, 4, stride=2, padding=1),
                       nn.ReLU(),
                       nn.Conv2d(32, 8, 4, stride=2),
                       nn.ReLU(),
                       nn.ConvTranspose2d(8, 8, 4, stride=2),
                       nn.ReLU(),
                       nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1),
                       nn.Sigmoid())

    with torch.no_grad():
        for p, p_ in zip(s.param(), s_.parameters()):
            p.copy_(p_.data.clone())

    s_.to('cuda')
    s.cuda()
    opt = Adam(s, lr=1e-3, device='cuda')
    torch_opt = torch.optim.Adam(s_.parameters(), lr=1e-3)
    noisy_imgs_1, noisy_imgs_2 = torch.load('/home/pc/PycharmProjects/pythonProject/data/train_data.pkl')

    l2_loss = MSE()
    torch_l2_loss = nn.MSELoss()

    norm_input = (noisy_imgs_1 / 255).split(128)
    norm_target = (noisy_imgs_2 / 255).split(128)

    for inputs, targets in tqdm(zip(norm_input, norm_target), total=len(norm_input)):
        torch.set_grad_enabled(False)
        inp = inputs.to('cuda')
        target = targets.to('cuda')
        opt.zero_grad()
        output = s(inp)
        l = l2_loss(target, output)
        l.creator.backward()
        l2_loss.clear_saved_tensors()
        opt.step()

        torch.set_grad_enabled(True)
        torch_opt.zero_grad()
        output = s_(inp)
        l = torch_l2_loss(target, output)
        l.backward()
        torch_opt.step()
        for p, p_ in zip(s.param(), s_.parameters()):
            assert torch.allclose(p, p_), "Parameters diverged for Adam"


if __name__ == '__main__':
    main()

