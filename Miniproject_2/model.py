from torch import empty
from torch.nn.functional import fold, unfold
import pickle

class Model():
    def __init__(self):
        self.net = Sequential(Conv2d(3, 16, 4, stride=2, padding=1),
                               ReLU(),
                               Conv2d(16, 32, 4, stride=2),
                               ReLU(),
                               TransposeConv2d(32, 16, 4, stride=2),
                               ReLU(),
                               TransposeConv2d(16, 3, 4, stride=2, padding=1),
                               Sigmoid())
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
        if self.cuda:
            self.optimizer = SGD(self.net, lr=0.5, momentum=0.9, device='cuda')
        else:
            self.optimizer = SGD(self.net, lr=0.5, momentum=0.9, device='cpu')
        self.loss = MSE()

    def load_pretrained_model(self):
        from pathlib import Path
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.net.load(model_path)
        if self.cuda:
            self.net.cuda()

    def train(self, train_input, train_target, num_epochs):

        norm_input = (train_input / 255).split(self.batch_size)
        norm_target = (train_target / 255).split(self.batch_size)

        for _ in range(num_epochs):
            for inputs, targets in zip(norm_input, norm_target):
                if self.cuda:
                    inputs = inputs.to('cuda')
                    targets = targets.to('cuda')
                self.optimizer.zero_grad()
                output = self.net(inputs)
                l = self.loss(targets, output)
                l.creator.backward()
                self.optimizer.step()
                # Necessary because loss is not part of the model
                self.loss.clear_saved_tensors()

    def predict(self, test_input):
        if self.cuda:
            out = (self.net(test_input.to('cuda') / 255) * 255)
            return out.clip_(0, 255)
        else:
            return (self.net(test_input / 255) * 255).clip_(0, 255)

    def save(self, checkpointpath):
        self.net.save(checkpointpath)

class Module(object):
    def __init__(self):
        self.xs = []

    def clear_saved_tensors(self):
        self.xs = []

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplemented

    def param(self):
        return []

    def cuda(self):
        pass

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, input):
        self.xs.append(input)
        out = input.clip(0)
        out.creator = self
        out.input_index = len(self.xs) - 1
        return out

    def backward(self, gradwrtoutput):
        x = self.xs[gradwrtoutput.input_index]
        dldx = x.clone()
        dldx[dldx <= 0] = 0
        dldx[dldx > 0] = 1
        if hasattr(x, 'creator'):
            grad = dldx * gradwrtoutput
            grad.input_index = gradwrtoutput.input_index
            x.creator.backward(grad)

class Sequential(Module):

    def __init__(self, *modules):
        self.modules = modules

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, input):
        x = input
        for m in self.modules:
            x = m(x)

        x.creator = self
        return x

    def backward(self, gradwrtoutput):
        self.modules[-1].backward(gradwrtoutput)

    def param(self):
        params = []
        for m in self.modules:
            params += m.param()
        return params

    def cuda(self):
        for m in self.modules:
            m.cuda()

    def clear_saved_tensors(self):
        for m in self.modules:
            m.clear_saved_tensors()

    def save(self, checkpointpath):
        with open(checkpointpath, 'wb') as file:
            pickle.dump([p.cpu() for p in self.param()], file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, checkpointpath):
        with open(checkpointpath, 'rb') as file:
            checkpoint = pickle.load(file)
        params = self.param()
        for p, p_ in zip(params, checkpoint):
            p.copy_(p_)
            p.grad = None
            p.creator = None

class MSE(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, y, y_hat):
        e = y_hat - y
        if hasattr(y_hat, 'creator'):
            e.creator = y_hat.creator

        self.xs.append(e)
        out = (e ** 2).mean()
        out.creator = self
        return out

    def backward(self, *gradwrtoutput):
        for i, e in enumerate(self.xs):
            n = e.numel()
            grad = 2/n * e
            grad.input_index = i
            if hasattr(e, 'creator'):
                e.creator.backward(grad)

class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, dilation = 1):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        if type(kernel_size) == int:
            self.kernel_size = (kernel_size,kernel_size)
        else:
            self.kernel_size = kernel_size
        if type(stride) == int:
            self.stride = (stride,stride)
        else:
            self.stride = stride
        if type(padding) == int:
            self.padding = (padding,padding)
        else:
            self.padding = padding
        if type(dilation) == int:
            self.dilation = (dilation,dilation)
        else:
            self.dilation = dilation
        k = (1 / (in_channels * self.kernel_size[0] * self.kernel_size[1])) ** 0.5
        self.weight = empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-k, k)
        self.bias = empty(out_channels).uniform_(-k, k)
        self.xs = []
        self.x_shapes = []

    def cuda(self):
        self.weight = self.weight.to('cuda')
        self.bias = self.bias.to('cuda')

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, input):
        N = input.shape[0]
        self.x_shapes.append(input.shape)
        self.xs.append(input)
        unfolded = unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)

        wxb = self.weight.view(self.out_channels, -1) @ unfolded + self.bias.view(1, -1, 1)
        Hout = int((input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)/self.stride[0] + 1)
        out = wxb.view(N, self.out_channels , Hout , -1)
        out.creator = self
        out.input_index = len(self.xs) - 1
        return out

    def backward(self, gradwrtoutput):
        x, g = self.xs[gradwrtoutput.input_index], gradwrtoutput

        x_shape = x.shape
        x_u = unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
        unfold_grad = g.flatten(start_dim=-2).permute(1, 0, 2).reshape(self.out_channels, -1)

        x_reshape = x_u.permute(0, 2, 1).reshape(unfold_grad.shape[1], -1)
        grad = unfold_grad @ x_reshape
        if self.weight.grad is None:
            self.weight.grad = grad.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight.grad += grad.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        if self.bias.grad is None:
            self.bias.grad = g.transpose(0, 1).reshape(self.out_channels, -1).sum(axis = 1)
        else:
            self.bias.grad += g.transpose(0, 1).reshape(self.out_channels, -1).sum(axis = 1)

        gxr = self.weight.view(self.out_channels, -1).t() @ g.flatten(2)
        dl_dx = fold(gxr, x_shape[-2:], self.kernel_size, self.dilation, self.padding, self.stride)

        dl_dx.input_index = gradwrtoutput.input_index
        if hasattr(x, 'creator'):
            x.creator.backward(dl_dx)

    def param(self):
        return [self.weight, self.bias]

class Upsampling(Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, stride):
        super().__init__()
        self.upsample = TransposeConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        dilation=dilation, padding=padding, stride=stride)

    def backward(self, gradwrtoutput):
        self.upsample.backward(gradwrtoutput)

    def forward(self, input):
        return self.upsample(input)

    def __call__(self, *args, **kwargs):
        return self.forward(*args)


class TransposeConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, dilation = 1):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        if type(kernel_size) == int:
            self.kernel_size = (kernel_size,kernel_size)
        else:
            self.kernel_size = kernel_size
        if type(stride) == int:
            self.stride = (stride,stride)
        else:
            self.stride = stride
        if type(padding) == int:
            self.padding = (padding,padding)
        else:
            self.padding = padding
        if type(dilation) == int:
            self.dilation = (dilation,dilation)
        else:
            self.dilation = dilation

        k = (1 / (out_channels * self.kernel_size[0] * self.kernel_size[1])) ** 0.5
        self.weight = empty(in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-k, k)
        self.bias = empty(out_channels).uniform_(-k, k)

    def cuda(self):
        self.weight = self.weight.to('cuda')
        self.bias = self.bias.to('cuda')

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, input):
        B, C, H, W = input.shape
        self.xs.append(input)
        H_out = int((H - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + 1)
        W_out= int((W - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + 1)

        gxr = self.weight.view(self.in_channels, -1).t() @ input.flatten(2)
        out = fold(gxr, (H_out, W_out), self.kernel_size, self.dilation, self.padding, self.stride) + self.bias.view(1, -1, 1, 1)

        out.creator = self
        out.input_index = len(self.xs) - 1
        return out

    def backward(self, gradwrtoutput):
        x, g = self.xs[gradwrtoutput.input_index], gradwrtoutput

        unfolded_g = unfold(g, self.kernel_size, self.dilation, self.padding, self.stride).transpose(-1,-2)
        xr = x.flatten(start_dim=-2)

        if self.weight.grad is None:
            self.weight.grad = (xr @ unfolded_g).reshape(-1, self.in_channels, self.out_channels, *self.kernel_size).sum(axis=0)
        else:
            self.weight.grad += (xr @ unfolded_g).reshape(-1, self.in_channels, self.out_channels, *self.kernel_size).sum(axis=0)

        if self.bias.grad is None:
            self.bias.grad = g.transpose(0, 1).reshape(self.out_channels, -1).sum(axis = 1)
        else:
            self.bias.grad += g.transpose(0, 1).reshape(self.out_channels, -1).sum(axis = 1)

        w_f = self.weight.flatten(start_dim=-3)
        dL_dx = (w_f @ unfolded_g.transpose(-1, - 2)).reshape(-1,self.in_channels,*x.shape[-2:])

        dL_dx.input_index = gradwrtoutput.input_index

        if hasattr(x, 'creator'):
            x.creator.backward(dL_dx)

    def param(self):
        return [self.weight, self.bias]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, input):
        self.xs.append(input)
        out = input.sigmoid()
        out.creator = self
        out.input_index = len(self.xs) - 1
        return out

    def backward(self, gradwrtoutput):
        x, g = self.xs[gradwrtoutput.input_index], gradwrtoutput
        s = x.sigmoid()
        dldx = s * (1 - s) * g
        dldx.input_index = gradwrtoutput.input_index
        if hasattr(x, 'creator'):
            x.creator.backward(dldx)


class SGD(Module):
    def __init__(self, model, lr=1e-4, momentum=0.0, device='cpu'):
        self.params = model.param()
        self.had_one_step = False
        if momentum > 0:
            self.momentum = momentum
            self.momentum_tensors = []
            for p in self.params:
                self.momentum_tensors.append(empty(p.shape).fill_(0).to(device))
        else:
            self.momentum = None
            self.momentum_tensors = None

        self.model = model
        self.lr = lr

    def forward(self, *input):
        raise NotImplementedError

    def step(self):
        if self.momentum is None:
            for p in self.params:
                if p.grad is not None:
                    p.sub_(self.lr * p.grad)
        else:
            for i, p in enumerate(self.params):
                if p.grad is not None:
                    if self.had_one_step:
                        self.momentum_tensors[i] = self.momentum * self.momentum_tensors[i] + p.grad
                        p.sub_(self.lr * self.momentum_tensors[i])
                    else:
                        self.momentum_tensors[i] = p.grad
                        p.sub_(self.lr * self.momentum_tensors[i])
                        self.had_one_step = True

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None
        self.model.clear_saved_tensors()

    def param(self):
        return []

class Adam(Module):
    def __init__(self, model, lr=1e-3, betas = (0.9, 0.999), eps=1e-8, device='cpu'):
        self.eps = eps
        self.params = model.param()
        self.model = model
        self.b1, self.b2 = betas
        self.t = 0

        self.m = []
        self.v = []

        for p in self.params:
            self.m.append(empty(p.shape).fill_(0).to(device))
            self.v.append(empty(p.shape).fill_(0).to(device))

        self.lr = lr

    def forward(self, *input):
        raise NotImplementedError

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * p.grad
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (p.grad ** 2)
                m_hat = self.m[i] / (1 - self.b1 ** self.t)
                v_hat = self.v[i] / (1 - self.b2 ** self.t)
                p.sub_(self.lr * m_hat / (v_hat ** 0.5 + self.eps))

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None
        self.model.clear_saved_tensors()

    def param(self):
        return []