import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
    'half_identity' : lambda C, stride, affine: HalfIdentity(),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SEAttentionWeightedAggregation(nn.Module):
    
    def __init__(self, channels, num_inputs=2, reduction_ratio=4, dropout_p=0.3):
        super(SEAttentionWeightedAggregation, self).__init__()
        self.num_inputs = num_inputs
        self.channels = channels

        self.raw_weights = nn.Parameter(torch.ones(num_inputs))
        self.softmax = nn.Softmax(dim=-1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(  
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        normalized_weights = self.softmax(self.raw_weights)
        x_fused = torch.stack(inputs, dim=0) # (num_inputs, B, C, H, W)
        x_fused = torch.sum(x_fused * normalized_weights.view(-1, 1, 1, 1, 1), dim=0)

        squeezed = self.squeeze(x_fused).view(x_fused.size(0), self.channels)

        dynamic_k = self.excitation(squeezed)

        output = x_fused * dynamic_k.view(x_fused.size(0), self.channels, 1, 1)

        return output


#========== Ablation Study ==========

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out


class IdentityAdd(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class HalfIdentity(nn.Module):
    
    def forward(self, x):
        return x * 0.5


class LearnableWeightedAggregation(nn.Module):
    
    def __init__(self, num_inputs=2, C=None):
        super(LearnableWeightedAggregation, self).__init__()
        
        self.num_inputs = num_inputs
        self.C = C
        
        self.raw_weights = nn.Parameter(torch.ones(num_inputs))
        self.raw_k = nn.Parameter(torch.ones(1))
        
        self.softmax_fn = nn.Softmax(dim=-1)
        self.softplus_fn = nn.Softplus()

    def forward(self, x_in):
        x1, x2 = x_in
        normalized_weights = self.softmax_fn(self.raw_weights)
        w1_prime, w2_prime = normalized_weights[0], normalized_weights[1]
        weighted_sum = w1_prime * x1 + w2_prime * x2
        k = self.softplus_fn(self.raw_k)
        output = k * weighted_sum
        return output


class FusionOperationState:
    
    def __init__(self, fusion_function):
        self.buffer = None
        self.fusion_function = fusion_function


class FuseFirstInput(nn.Module):
    
    def __init__(self, state: FusionOperationState):
        super().__init__()
        self.state = state

    def forward(self, x):
        self.state.buffer = x
        return torch.zeros_like(x)


class FuseSecondInputAndApply(nn.Module):
    
    def __init__(self, state: FusionOperationState):
        super().__init__()
        self.state = state

    def forward(self, x2):
        x1 = self.state.buffer
        if x1 is None:
            raise ValueError("First input was not buffered. Check operation order.")
        
        result = self.state.fusion_function(x1, x2)
        
        self.state.buffer = None
        return result