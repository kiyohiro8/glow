import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class Glow(nn.Module):
    def __init__(self, params):
        super(Glow, self).__init__()
        self.flow = FlowNet(
            image_shape=params["image_shape"],
            hidden_features=params["hidden_features"],
            K=params["K"],
            L=params["L"],
            actnorm_scale=params["actnorm_scale"]
            )

    def forward(self, x):
        return self.inference(x)

    def inference(self, x):
        num_pixels = util.count_pixels(x)
        x = x + torch.normal(
            mean=torch.zeros_like(x),
            std=torch.ones_like(x) / 255)
        
        logdet = torch.zeros_like(x[:, 0, 0, 0])
        logdet += float(-np.log(256.) * num_pixels)

        z, objective = self.flow.encode(x, logdet)

        mean = torch.zeros_like(z, device=x.device)
        logs = torch.ones_like(z, device=x.device)

        objective += GaussianDiag.logp(mean, logs, z)
        nll = (- objective) / (np.log(2) * num_pixels)
        return z, nll

    def generate(self, z, eps_std):
        x = self.flow.decode(z, eps_std=eps_std)
        return x

    def initialize_actnorm(self, x):
        self.forward(x)



class FlowNet(nn.Module):
    def __init__(self, image_shape, hidden_features, K, L, actnorm_scale=1):
        super(FlowNet, self).__init__()
        self.layers = []
        H, W, C = image_shape

        for i in range(L):
            C = C * 4
            self.layers.append(SqueezeLayer(factor=2))
            for _ in range(K):
                self.layers.append(FlowStepLayer(in_features=C, hidden_features=hidden_features, actnorm_scale=actnorm_scale))
            if i < L -1:
                self.layers.append(Split2d(num_features=C))
                C = C // 2
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input):
        raise NotImplementedError

    def encode(self, x, logdet=0):
        for i, layer in enumerate(self.layers):
            x, logdet = layer(x, logdet, reverse=False)
        return x, logdet

    def decode(self, z, eps_std=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z = layer(z, eps_std=eps_std, reverse=True)
            else:
                z = layer(z, reverse=True)
        return z

class FlowStepLayer(nn.Module):
    def __init__(self, in_features, hidden_features, actnorm_scale=1.0):
        super(FlowStepLayer, self).__init__()
        self.actnorm = ActNorm2d(in_features, actnorm_scale)
        self.flow_permute = InvertibleConv1x1(
            in_features
        )
        self.flow_coupling = nn.Sequential(
            ActNormConv2d(in_features // 2, hidden_features),
            nn.ReLU(inplace=False),
            ActNormConv2d(hidden_features, hidden_features, kernel_size=[1, 1]),
            nn.ReLU(inplace=False),
            Conv2dZeros(hidden_features, in_features)
        )

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        z, logdet = self.actnorm(input, logdet, reverse=False)
        z, logdet = self.flow_permute(z, logdet, False)
        z1, z2 = util.split_feature(z, "split")
        h = self.flow_coupling(z1)
        shift, scale = util.split_feature(h, "cross")
        scale = torch.sigmoid(scale + 2)
        z2 = (z2 + shift) * scale
        logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat([z1, z2], dim=1)
        return z, logdet

    def reverse_flow(self, input, logdet):
        z1, z2 = util.split_feature(input, "split")
        h = self.flow_coupling(z1)
        shift, scale = util.split_feature(h, "cross")
        scale = torch.sigmoid(scale + 2)
        z2 = z2 / scale - shift
        logdet = - torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet

        z = torch.cat([z1, z2], dim=1)

        z, logdet = self.flow_permute(z, logdet, True)

        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet


class ActNorm2d(nn.Module):
    def __init__(self, in_features, scale=1):
        super(ActNorm2d, self).__init__()
        self.register_parameter(
            "bias",
            nn.Parameter(torch.zeros(1, in_features, 1, 1, device="cuda"))
        )
        self.register_parameter(
            "logs",
            nn.Parameter(torch.zeros(1, in_features, 1, 1, device="cuda"))
        )
        self.features = in_features
        self.scale = scale
        self.initialized = False

    def initialize(self, input):
        with torch.no_grad():
            bias = - torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.initialized = True

    def _center(self, input, reverse=False):
        if not reverse:
            return input + self.bias
        else:
            return input - self.bias

    def _scale(self, input, logdet=None, reverse=False):
        logs = self.logs
        num_pixels = util.count_pixels(input)
        if not reverse:
            output = input * torch.exp(logs)
        else:
            output = input * torch.exp(-logs)

        if logdet is not None:
            dlogdet = torch.sum(logs) * num_pixels
            if reverse:
                dlogdet = - dlogdet
            logdet = logdet + dlogdet
        return output, logdet

    def forward(self, input, logdet=None, reverse=False):
        if not self.initialized:
            self.initialize(input)
        if not reverse:
            output = self._center(input, reverse)
            output, logdet = self._scale(output, logdet, reverse)
        else:
            output, logdet = self._scale(input, logdet, reverse)
            output = self._center(output, reverse)
        return output, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_features):
        super(InvertibleConv1x1, self).__init__()
        w_shape = [num_features, num_features]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        np_p, np_l, np_u = scipy.linalg.lu(w_init)
        np_s = np.diag(np_u)
        np_sign_s = np.sign(np_s)
        np_log_s = np.log(np.abs(np_s))
        np_u = np.triu(np_u, k=1)
        l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
        eye = np.eye(*w_shape, dtype=np.float32)

        self.register_buffer("p", torch.Tensor(np_p.astype(np.float32)))
        self.register_buffer("sign_s", torch.Tensor(np_sign_s.astype(np.float32)))
        self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
        self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
        self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
        self.l_mask = torch.Tensor(l_mask)
        self.eye = torch.Tensor(eye)

        self.w_shape = w_shape
        self.LU = True
        self.to(torch.device("cuda"))

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        num_pixels = util.count_pixels(input)
        self.p = self.p.to(input.device)
        self.sign_s = self.sign_s.to(input.device)
        self.l_mask = self.l_mask.to(input.device)
        self.eye = self.eye.to(input.device)
        l = self.l * self.l_mask + self.eye
        u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
        dlogdet = torch.sum(self.log_s) * num_pixels

        if not reverse:
            w = torch.matmul(self.p, torch.matmul(l, u))
        else:
            l = torch.inverse(l.double()).float()
            u = torch.inverse(u.double()).float()
            w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
        return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            x = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return x, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class ActNormConv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = ActNormConv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", weight_std=0.05):
        padding = ActNormConv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=False)
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        self.actnorm = ActNorm2d(out_channels)

    def forward(self, input):
        x = super().forward(input)
        x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", logscale_factor=3):
        padding = ActNormConv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = self._squeeze2d(input, self.factor)
            return output, logdet
        else:
            output = self._unsqueeze2d(input, self.factor)
            return output, logdet

    def _squeeze2d(self, input, factor=2):
        if factor == 1:
            return input
        B, C, H, W = input.size()
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)
        return x

    def _unsqueeze2d(self, input, factor=2):
        if factor == 1:
            return input
        B, C, H, W = input.size()
        assert C % (factor) == 0, "{}".format(C)
        x = input.view(B, C // factor, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // (factor), H * factor, W * factor)
        return x


class Split2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv = Conv2dZeros(num_features // 2, num_features)

    def split2d_prior(self, z):
        h = self.conv(z)
        return util.split_feature(h, "cross")

    def forward(self, input, logdet=0., reverse=False, eps_std=None):
        if not reverse:
            z1, z2 = util.split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = GaussianDiag.logp(mean, logs, z2) + logdet
            return z1, logdet
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = GaussianDiag.sample(mean, logs, eps_std)
            z = torch.cat([z1, z2], dim=1)
            return z, logdet


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return torch.sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps