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
        # 단순히 x를 반환 (자동으로 합연산이 될 것임)
        return x


class BilinearDownsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(BilinearDownsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # Bilinear downsampling (upscale then downsample, or directly downsample)
        return F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear', align_corners=False)

class HalfIdentity(nn.Module):
    def forward(self, x):
        return x * 0.5

class AvgPoolDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)



class LearnableWeightedAggregation(nn.Module):
    def __init__(self, num_inputs=2, C=None):
        super(LearnableWeightedAggregation, self).__init__()
        
        self.num_inputs = num_inputs
        self.C = C
        
        # 원래 기본 초기화
        self.raw_weights = nn.Parameter(torch.ones(num_inputs))  # softmax 통과 예정
        self.raw_k = nn.Parameter(torch.ones(1))  # softplus 통과 예정
        
        # Softmax와 Softplus
        self.softmax_fn = nn.Softmax(dim=-1)
        self.softplus_fn = nn.Softplus()

    def forward(self, x_in):
        x1, x2 = x_in
        normalized_weights = self.softmax_fn(self.raw_weights)  # w' = softmax(raw_weights)
        w1_prime, w2_prime = normalized_weights[0], normalized_weights[1]
        weighted_sum = w1_prime * x1 + w2_prime * x2
        k = self.softplus_fn(self.raw_k)  # k = softplus(raw_k)
        output = k * weighted_sum
        return output


class SEAttentionWeightedAggregation(nn.Module):
    """
    Squeeze-and-Excitation 기반의 동적 어텐션을 사용한 가중 집계 모듈.
    - 입력들을 학습 가능한 가중치(w')로 합산 (Fusion).
    - 합산된 결과에 채널별 동적 스케일(k, 어텐션)을 적용.
    """
    def __init__(self, channels, num_inputs=2, reduction_ratio=4, dropout_p=0.3):
        """
        Args:
            channels (int): 입력 피처맵의 채널 수.
            num_inputs (int): 입력 텐서의 개수.
            reduction_ratio (int): SE 모듈의 bottleneck 채널 감소 비율.
            dropout_p (float): Dropout 확률.
        """
        super(SEAttentionWeightedAggregation, self).__init__()
        self.num_inputs = num_inputs
        self.channels = channels

        # 1. Fusion 가중치 (w'): 입력들을 합치기 위한 학습 가능한 파라미터
        self.raw_weights = nn.Parameter(torch.ones(num_inputs))
        self.softmax = nn.Softmax(dim=-1)

        # 2. Dynamic Scaler (k): SE 모듈로 동적 스케일링 구현
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(  
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        """
        Args:
            inputs (list of torch.Tensor): 집계할 텐서들의 리스트. e.g., [x1, x2]

        Returns:
            torch.Tensor: 최종적으로 집계되고 스케일링된 텐서.
        """
        # Step 1: Fusion - 가중합 계산
        # raw_weights를 softmax에 통과시켜 합이 1인 가중치 w' 생성
        normalized_weights = self.softmax(self.raw_weights)
        # 각 입력에 가중치를 곱하여 합산
        x_fused = torch.stack(inputs, dim=0) # (num_inputs, B, C, H, W)
        x_fused = torch.sum(x_fused * normalized_weights.view(-1, 1, 1, 1, 1), dim=0)

        # Step 2: Squeeze - 채널별 정보 요약
        # (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        squeezed = self.squeeze(x_fused).view(x_fused.size(0), self.channels)

        # Step 3: Excitation - 채널별 중요도(동적 k) 계산
        # (B, C) -> (B, C)
        dynamic_k = self.excitation(squeezed)

        # Step 4: Rescale - 입력에 채널별 중요도를 곱하여 스케일링
        # (B, C) -> (B, C, 1, 1)로 차원 확장 후 곱셈
        output = x_fused * dynamic_k.view(x_fused.size(0), self.channels, 1, 1)

        return output
    


class FusionOperationState:
    def __init__(self, fusion_function):
        self.buffer = None  # 첫 번째 입력을 임시로 저장할 공간
        self.fusion_function = fusion_function # max, multiply 등 수행할 연산

# 클래스 2: '짝'의 첫 번째 - 입력을 받아 저장소에 넣고 0을 반환
class FuseFirstInput(nn.Module):
    def __init__(self, state: FusionOperationState):
        super().__init__()
        self.state = state

    def forward(self, x):
        # 1. 첫 번째 입력을 공유 저장소에 저장합니다.
        self.state.buffer = x
        # 2. 모델의 기존 '+' 연산에 영향을 주지 않도록 0을 반환합니다.
        return torch.zeros_like(x)

# 클래스 3: '짝'의 두 번째 - 새 입력을 받아 저장된 값과 함께 최종 연산 수행
class FuseSecondInputAndApply(nn.Module):
    def __init__(self, state: FusionOperationState):
        super().__init__()
        self.state = state

    def forward(self, x2):
        # 1. 공유 저장소에서 먼저 들어온 입력(x1)을 가져옵니다.
        x1 = self.state.buffer
        if x1 is None:
            # 혹시 모를 오류 방지
            raise ValueError("First input was not buffered. Check operation order.")
        
        # 2. 미리 정의된 fusion_function(max 등)을 수행하고 결과를 반환합니다.
        result = self.state.fusion_function(x1, x2)
        
        # 3. 다음 계산을 위해 저장소를 비워줍니다 (메모리 누수 방지).
        self.state.buffer = None
        return result