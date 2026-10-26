import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Channel(nn.Module):
	def __init__(self, channels):
		super(Channel, self).__init__()
		self.maxpool = nn.AdaptiveMaxPool2d(3)
		self.mlp = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=3, groups=channels) for _ in range(12)])

	def forward(self, x):
		avg_out = self.maxpool(x)
		output = [nn.Sigmoid()(nn.LeakyReLU()(mlp(avg_out))) for mlp in self.mlp]
		add = nn.LeakyReLU()(sum(output))
		out = nn.LeakyReLU()(add * x)

		return out
        
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    assert num_channels % groups == 0, "channel number can not be divisible by groups"
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x

class Channel(nn.Module):
    def __init__(self, channels, groups=2):
        super(Channel, self).__init__()
        self.groups = groups

        self.maxpool = nn.AdaptiveMaxPool2d(3)
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, groups=channels),
                nn.LeakyReLU(inplace=True),
                nn.Sigmoid()
            ) for _ in range(12)
        ])
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        avg_out = self.maxpool(x)
        output = [mlp(avg_out) for mlp in self.mlp]
        add = self.leaky_relu(sum(output))
        shuffled = channel_shuffle(add, self.groups)
        out = self.leaky_relu(shuffled * x)

        return out

class Channel(nn.Module):
    def __init__(self, channels):
        super(Channel, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1, groups=channels) for _ in range(12)])
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, groups=channels)

    def forward(self, x):
        avg_out = self.maxpool(x)
        output = [nn.Sigmoid()(nn.LeakyReLU()(mlp(avg_out))) for mlp in self.mlp]
        add = nn.LeakyReLU()(sum(output))

        batch_size, channels, height, width = add.shape
        interleaved = torch.empty(batch_size, channels * 2, height, width, device=x.device)

        # 交替嵌入通道
        interleaved[:, 0::2, :, :] = add
        interleaved[:, 1::2, :, :] = avg_out

        result = self.reduce(interleaved)
        out = nn.LeakyReLU()(result * x)

        return out

#if __name__ == "__main__":


    #x = torch.randn(1, 32, 64, 64)
    #model = channel(32)
    #print(model(x).shape)

class Spatial(nn.Module):
	def __init__(self, channels):
		super(Spatial, self).__init__()
		self.project_in = nn.Conv2d(channels, channels*3, kernel_size=1)
		self.dwconv = nn.Conv2d(channels*3, channels*3, kernel_size=3, padding=1, groups=channels*3)
		self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

	def forward(self, x):
		x = self.project_in(x)
		x1, x2, x3 = self.dwconv(x).chunk(3, dim=1)
		a = nn.LeakyReLU()(x1 * x2)
		b = nn.LeakyReLU()(a * x3)
		out = self.project_out(b)
		out = nn.LeakyReLU()(out)
		
		return out

class MLP(nn.Module):
	def __init__(self, channels):
		super(MLP, self).__init__()
		self.body = nn.Sequential(
				nn.Conv2d(channels, channels, kernel_size=1),
				nn.LeakyReLU(),
				nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, groups=channels)
				)

	def forward(self, x):

		return self.body(x)

class FeedForward(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.project_in = nn.Conv2d(channels, channels*2, kernel_size=1)
        self.dwconv = nn.Conv2d(channels*2, channels*2, kernel_size=3, stride=1, padding=1, groups=channels*2)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class OverlapPatchEmbed2(nn.Module):
    def __init__(self, in_c=1, embed_dim=48, bias=False):
        super(OverlapPatchEmbed2, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.MaxPool2d(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self, x):
        return self.body(x)

class Transformer(nn.Module):
	def __init__(self, channels):
		super(Transformer, self).__init__()
		self.norm1 = LayerNorm2d(channels)
		self.norm2 = LayerNorm2d(channels)

		self.spatial_blocks = Spatial(channels)
		self.channel_blocks = Channel(channels)
		self.drop = nn.Dropout2d(0.1)
		self.mlp = FeedForward(channels)

	def forward(self, img):
		x = self.norm1(img)
		x_1 = self.spatial_blocks(x)
		x_2 = nn.LeakyReLU()(x_1)
		x_3 = self.channel_blocks(x_2)
		x_4 = nn.LeakyReLU()(x_3)
		x_4 = self.drop(x_4)
		y = x_4 + img

		y_1 = self.norm2(y)
		y_2 = self.mlp(y_1)
		out = y_2 + y

		return out
