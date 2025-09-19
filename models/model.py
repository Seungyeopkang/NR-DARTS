import torch
import torch.nn as nn
import sys
from operations import *
from utils import drop_path
from genotypes import Genotype


class SearchCell(nn.Module):
    def __init__(
        self, genotype, 
        C_prev_prev, C_prev, C, 
        reduction, reduction_prev,
        affine=True, drop_path_prob=0.0  
    ):
        super(SearchCell, self).__init__()

        self.drop_path_prob = drop_path_prob

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._compile(C, op_names, indices, concat, reduction, affine)

        self.node_weights = nn.Parameter(torch.ones(len(op_names) // 2))

    def _compile(self, C, op_names, indices, concat, reduction, affine):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, affine=affine)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob, self.training)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob, self.training)

            s = h1 + h2
            s = self.node_weights[i] * s
            states.append(s)

        return torch.cat([states[i] for i in self._concat], dim=1)

    def arch_parameters(self):
        return [self.node_weights]

class WeightedBranchNormalCell(nn.Module):
    def __init__(
        self, genotype, 
        C_prev_prev, C_prev, C, 
        reduction, reduction_prev,
        affine=True, drop_path_prob=0.0  
    ):
        super(WeightedBranchNormalCell, self).__init__()

        self.drop_path_prob = drop_path_prob
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._compile(C, op_names, indices, concat, reduction, affine)

        # 각 op(x1), op(x2)에 weight 따로
        self.branch_weights = nn.Parameter(torch.ones(len(op_names)))

    def _compile(self, C, op_names, indices, concat, reduction, affine):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, affine=affine)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            idx1 = self._indices[2 * i]
            idx2 = self._indices[2 * i + 1]
            h1 = states[idx1]
            h2 = states[idx2]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob, self.training)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob, self.training)

            # branch별 개별 가중치 적용
            w1 = self.branch_weights[2 * i]
            w2 = self.branch_weights[2 * i + 1]
            s = w1 * h1 + w2 * h2
            states.append(s)

        return torch.cat([states[i] for i in self._concat], dim=1)

    def arch_parameters(self):
        return [self.branch_weights]

class WeightedBranchCell(nn.Module):
    def __init__(
        self, genotype, 
        C_prev_prev, C_prev, C, 
        reduction, reduction_prev,
        affine=True, drop_path_prob=0.0  
    ):
        super(WeightedBranchCell, self).__init__()

        self.drop_path_prob = drop_path_prob
        self.reduction = reduction  # reduction 여부 저장

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._compile(C, op_names, indices, concat, reduction, affine)

        # ✅ normal cell일 때만 branch_weights 생성
        if not reduction:
            self.branch_weights = nn.Parameter(torch.ones(len(op_names)))
        else:
            self.branch_weights = None

    def _compile(self, C, op_names, indices, concat, reduction, affine):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, affine=affine)
            self._ops.append(op)
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            idx1 = self._indices[2 * i]
            idx2 = self._indices[2 * i + 1]
            h1 = self._ops[2 * i](states[idx1])
            h2 = self._ops[2 * i + 1](states[idx2])

            if self.training and drop_prob > 0.:
                if not isinstance(self._ops[2 * i], Identity):
                    h1 = drop_path(h1, drop_prob, self.training)
                if not isinstance(self._ops[2 * i + 1], Identity):
                    h2 = drop_path(h2, drop_prob, self.training)

            if self.branch_weights is not None:
                # ✅ normal cell: 가중치 적용
                w1 = self.branch_weights[2 * i]
                w2 = self.branch_weights[2 * i + 1]
                s = w1 * h1 + w2 * h2
            else:
                # ✅ reduction cell: 그냥 더함
                s = h1 + h2

            states.append(s)

        return torch.cat([states[i] for i in self._concat], dim=1)

    def arch_parameters(self):
        return [self.branch_weights] if self.branch_weights is not None else []

class NormalOnlyWeightedSearchCell(nn.Module):
    def __init__(
        self, genotype, 
        C_prev_prev, C_prev, C, 
        reduction, reduction_prev,
        affine=True, drop_path_prob=0.0  
    ):
        super(NormalOnlyWeightedSearchCell, self).__init__()

        self.drop_path_prob = drop_path_prob
        self.reduction = reduction  # <-- normal/reduction 구분 저장

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._compile(C, op_names, indices, concat, reduction, affine)

        if not reduction:
            self.node_weights = nn.Parameter(torch.ones(len(op_names) // 2))
        else:
            self.node_weights = None  # reduction은 가중치 X

    def _compile(self, C, op_names, indices, concat, reduction, affine):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, affine=affine)
            self._ops.append(op)
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            if self.training and drop_prob > 0.:
                if not isinstance(op1, nn.Identity):
                    h1 = drop_path(h1, drop_prob, self.training)
                if not isinstance(op2, nn.Identity):
                    h2 = drop_path(h2, drop_prob, self.training)

            s = h1 + h2
            if not self.reduction:  # normal cell일 때만 weight 적용
                s = self.node_weights[i] * s

            states.append(s)

        return torch.cat([states[i] for i in self._concat], dim=1)

    def arch_parameters(self):
        return [self.node_weights] if self.node_weights is not None else []

class TrainCell(nn.Module):
    def __init__(
        self, genotype, 
        C_prev_prev, C_prev, C, 
        reduction, reduction_prev, 
        affine=True, drop_path_prob=0.0
    ):
        super().__init__()
        self.drop_path_prob = drop_path_prob

        # 이전 셀이 reduction일 경우 다운샘플링
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)

        # 연산 정의
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        self._indices = indices

        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, affine=affine)
            self._ops += [op]

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob, self.training)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob, self.training)

            s = h1 + h2
            states.append(s)

        return torch.cat([states[i] for i in self._concat], dim=1)

class TrainK1Cell(nn.Module):
    def __init__(
        self, genotype, 
        C_prev_prev, C_prev, C, 
        reduction, reduction_prev, 
        affine=True, drop_path_prob=0.0
    ):
        super().__init__()
        self.drop_path_prob = drop_path_prob
        self.reduction = reduction

        # 이전 셀이 reduction일 경우 다운샘플링
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)

        # K=1 탐색된 genotype에서 연산 정의
        if reduction:
            ops_with_indices = genotype.reduce
            concat = genotype.reduce_concat
        else:
            ops_with_indices = genotype.normal
            concat = genotype.normal_concat

        # K=1에서는 각 intermediate node마다 하나의 연산만 있음
        self._steps = len(ops_with_indices)
        
        # concat 인덱스를 실제 존재하는 상태에 맞게 조정
        expected_states = 2 + self._steps  # s0, s1 + intermediate nodes
        valid_concat = [i for i in concat if i < expected_states]
        if not valid_concat:
            valid_concat = list(range(2, expected_states))  # 모든 intermediate nodes
            
        self._concat = valid_concat
        self.multiplier = len(valid_concat)

        # 각 step별로 하나의 연산만 구성
        self._ops = nn.ModuleList()
        self._indices = []

        for i, (op_name, input_idx) in enumerate(ops_with_indices):
            # 입력 인덱스가 유효한 범위인지 확인하고 조정
            max_input_idx = 1 + i  # step i에서 사용 가능한 최대 인덱스
            safe_input_idx = min(input_idx, max_input_idx)
            
            stride = 2 if reduction and safe_input_idx < 2 else 1
            op = OPS[op_name](C, stride, affine=affine)
            
            self._ops.append(op)
            self._indices.append(safe_input_idx)

    def forward(self, s0, s1, drop_prob):
        # preprocessing
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        # 각 intermediate node 계산 (K=1이므로 하나의 연산만)
        for i in range(self._steps):
            # 입력 선택
            input_idx = self._indices[i]
            h = states[input_idx]
            
            # 연산 적용
            op = self._ops[i]
            h = op(h)

            # Drop path 적용 (학습 중에만)
            if self.training and drop_prob > 0.:
                if not isinstance(op, Identity):
                    h = drop_path(h, drop_prob, self.training)

            # 새로운 intermediate node 추가
            states.append(h)

        # concat으로 최종 출력 생성
        return torch.cat([states[i] for i in self._concat], dim=1)

class LearnableFusionCell(nn.Module):
    def __init__(
        self, genotype, 
        C_prev_prev, C_prev, C, 
        reduction, reduction_prev, 
        affine=True, drop_path_prob=0.0,
        # ================== 추가된 인자 ==================
        pruned_nodes_info=None, # (cell_idx, node_idx) 튜플을 키로 가지는 딕셔너리
        cell_idx=None           # 현재 셀의 전체 모델 내 인덱스
        # =================================================
    ):
        super().__init__()
        self.drop_path_prob = drop_path_prob
        self.cell_idx = cell_idx # 현재 셀의 인덱스 저장

        # 프루닝된 노드 정보를 저장 (현재 셀에 해당하는 노드만)
        self.pruned_nodes_local_info = {}
        if pruned_nodes_info:
            for c_idx, n_idx in pruned_nodes_info.keys(): # 딕셔너리 키(튜플)만 순회
                if c_idx == self.cell_idx:
                    self.pruned_nodes_local_info[n_idx] = True # {node_idx: True} 형태로 저장

        # ================== Aggregation 인스턴스 초기화 ==================
        self.learnable_aggregators = nn.ModuleDict()
        # DARTS 셀의 각 intermediate node는 2개의 입력을 가집니다 (h1, h2)
        # _steps 만큼의 intermediate node가 있습니다.
        # 각 intermediate node i에 대해, 해당 노드가 프루닝되었다면 LearnableWeightedAggregation 인스턴스 생성
        # 여기서 i는 0부터 self._steps-1 까지의 값입니다.
        # 주의: _steps는 _ops가 정의된 후에 알 수 있으므로, 임시로 genotype 길이 사용
        # _steps가 정확히 계산된 후 이 루프를 다시 돌리거나, forward에서 조건을 더 명확히 해야 할 수 있습니다.
        # 그러나 현재 _steps는 __init__ 마지막 부분에서 설정되므로, 여기서 len(genotype...)을 쓰는 것이 맞습니다.
        for i in range(len(genotype.normal) // 2 if not reduction else len(genotype.reduce) // 2): 
             # genotype.normal 또는 reduce는 (op_name, index) 쌍이 2개씩 있으므로 // 2
            if i in self.pruned_nodes_local_info:
                # LearnableWeightedAggregation은 입력 채널 C를 받도록 수정될 수 있음
                self.learnable_aggregators[str(i)] = LearnableWeightedAggregation(C=C, num_inputs=2) # C는 현재 셀의 출력 채널
        # =================================================================

        # 이전 셀이 reduction일 경우 다운샘플링
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)

        # 연산 정의
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._steps = len(op_names) // 2 # 중간 노드의 개수
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        self._indices = indices # 각 op의 입력 소스 인덱스

        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, affine=affine)
            self._ops += [op]

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        for i in range(self._steps): # 각 중간 노드 (i)에 대해
            h1_input = states[self._indices[2 * i]]    # 첫 번째 입력 소스에서 피처맵 가져옴
            h2_input = states[self._indices[2 * i + 1]] # 두 번째 입력 소스에서 피처맵 가져옴
            
            # ================== Aggregation 로직 변경 (핵심 수정 부분) ==================
            if i in self.pruned_nodes_local_info:
                # 현재 노드(i)가 프루닝된 노드인 경우 LearnableWeightedAggregation 사용
                # 이 경우, 원래의 op1, op2 연산을 건너뛰고 LWA에 직접 입력 전달
                s = self.learnable_aggregators[str(i)]([h1_input, h2_input]) 
            else:
                # 프루닝되지 않은 노드는 기존 방식 (genotype에 따라 선택된 op1, op2 연산 후 단순 합산)
                op1 = self._ops[2 * i]      # 첫 번째 입력에 적용할 연산
                op2 = self._ops[2 * i + 1]   # 두 번째 입력에 적용할 연산

                h1 = op1(h1_input)
                h2 = op2(h2_input)

                if self.training and drop_prob > 0.:
                    # drop_path_prob 적용 (Identity 연산이 아닐 경우)
                    if not isinstance(op1, Identity):
                        h1 = drop_path(h1, drop_prob, self.training)
                    if not isinstance(op2, Identity):
                        h2 = drop_path(h2, drop_prob, self.training)
                
                s = h1 + h2 # 기존 방식: 연산 결과 합산
            # =========================================================
            
            states.append(s)

        return torch.cat([states[i] for i in self._concat], dim=1)

class SEFusionCell(nn.Module):
    def __init__(
        self, genotype,
        C_prev_prev, C_prev, C,
        reduction, reduction_prev,
        affine=True, drop_path_prob=0.0,
        pruned_nodes_info=None,
        cell_idx=None,
        dropout_p=0.3
    ):
        super().__init__()
        self.drop_path_prob = drop_path_prob
        self.cell_idx = cell_idx

        self.pruned_nodes_local_info = {}
        if pruned_nodes_info:
            for c_idx, n_idx in pruned_nodes_info.keys():
                if c_idx == self.cell_idx:
                    self.pruned_nodes_local_info[n_idx] = True

        # <<<<<<<<<<<<<<<< 핵심 수정 부분 >>>>>>>>>>>>>>>>
        self.learnable_aggregators = nn.ModuleDict()
        num_intermediate_nodes = len(genotype.normal) // 2 if not reduction else len(genotype.reduce) // 2
        for i in range(num_intermediate_nodes):
            if i in self.pruned_nodes_local_info:
                # SEAttentionWeightedAggregation 모듈을 사용하도록 변경
                self.learnable_aggregators[str(i)] = SEAttentionWeightedAggregation(
                    num_inputs=2,
                    channels=C,
                    reduction_ratio=4 # 또는 다른 값으로 설정
                )
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        self._indices = indices

        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, affine=affine)
            self._ops.append(op)

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        for i in range(self._steps):
            h1_input = states[self._indices[2 * i]]
            h2_input = states[self._indices[2 * i + 1]]

            if i in self.pruned_nodes_local_info:
                s = self.learnable_aggregators[str(i)]([h1_input, h2_input])
            else:
                op1 = self._ops[2 * i]
                op2 = self._ops[2 * i + 1]
                h1 = op1(h1_input)
                h2 = op2(h2_input)
                if self.training and drop_prob > 0.:
                    if not isinstance(op1, Identity):
                        h1 = drop_path(h1, drop_prob, self.training)
                    if not isinstance(op2, Identity):
                        h2 = drop_path(h2, drop_prob, self.training)
                s = h1 + h2
            states.append(s)

        return torch.cat([states[i] for i in self._concat], dim=1)


class PrunedSECell_Fixed(nn.Module):
    """
    가지치기된 셀 - 스칼라 파라미터 고정 버전
    - 가지치기된 노드: operation 삭제 + SE Attention으로 교체
    - 기존 노드: 스칼라 가중치 1로 고정 (학습 안됨)
    - SE 모듈만 fine-tuning에 참여
    """
    def __init__(
        self, genotype,
        C_prev_prev, C_prev, C,
        reduction, reduction_prev,
        affine=True, drop_path_prob=0.0,
        pruned_nodes_info=None,
        cell_idx=None,
        dropout_p=0.3
    ):
        super(PrunedSECell_Fixed, self).__init__()
        
        self.drop_path_prob = drop_path_prob
        self.reduction = reduction
        self.cell_idx = cell_idx
        
        # 가지치기 정보 처리 (이 셀에 해당하는 노드들만)
        self.pruned_nodes_local = set()
        if pruned_nodes_info:
            for (c_idx, n_idx) in pruned_nodes_info.keys():
                if c_idx == self.cell_idx:
                    self.pruned_nodes_local.add(n_idx)
        
        # Preprocessing layers (기존과 동일)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)
        
        # Genotype 정보 추출
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        
        self._compile(C, op_names, indices, concat, reduction, affine)
        
        # 스칼라 가중치 (기존 호환성 위해 유지하지만 고정값)
        if not reduction:
            # 고정된 스칼라 가중치 (학습 안됨)
            self.register_buffer('node_weights', torch.ones(len(op_names) // 2))
        else:
            self.node_weights = None
        
        # SE Attention 모듈들 (가지치기된 노드에만)
        self.se_modules = nn.ModuleDict()
        for node_idx in self.pruned_nodes_local:
            self.se_modules[str(node_idx)] = SEAttentionWeightedAggregation(
                channels=C,
                num_inputs=2,
                reduction_ratio=4,
                dropout_p=dropout_p
            )
    
    def _compile(self, C, op_names, indices, concat, reduction, affine):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, affine=affine)
            self._ops.append(op)
        self._indices = indices
    
    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        for i in range(self._steps):
            h1_input = states[self._indices[2 * i]]
            h2_input = states[self._indices[2 * i + 1]]
            
            if i in self.pruned_nodes_local:
                # 가지치기된 노드: operation 삭제, SE로 직접 처리
                s = self.se_modules[str(i)]([h1_input, h2_input])
            else:
                # 기존 노드: operation 수행 + 고정된 스칼라 가중치
                op1 = self._ops[2 * i]
                op2 = self._ops[2 * i + 1]
                h1 = op1(h1_input)
                h2 = op2(h2_input)
                
                if self.training and drop_prob > 0.:
                    if not isinstance(op1, nn.Identity):
                        h1 = drop_path(h1, drop_prob, self.training)
                    if not isinstance(op2, nn.Identity):
                        h2 = drop_path(h2, drop_prob, self.training)
                
                s = h1 + h2
                if not self.reduction:
                    # 고정된 가중치 (1.0) 적용
                    s = self.node_weights[i] * s
            
            states.append(s)
        
        return torch.cat([states[i] for i in self._concat], dim=1)
    
    def arch_parameters(self):
        # SE 모듈의 파라미터들만 반환 (스칼라 가중치는 고정)
        params = []
        for se_module in self.se_modules.values():
            params.extend(list(se_module.parameters()))
        return params


class PrunedSECell_OriginalWeight(nn.Module):
    """
    가지치기된 셀 - 기존 학습된 가중치 고정 버전
    - 가지치기된 노드: operation 삭제 + SE Attention으로 교체
    - 기존 노드: 기존 학습된 스칼라 가중치로 고정 (학습 안됨)
    - SE 모듈만 fine-tuning에 참여
    """
    def __init__(
        self, genotype,
        C_prev_prev, C_prev, C,
        reduction, reduction_prev,
        affine=True, drop_path_prob=0.0,
        pruned_nodes_info=None,
        cell_idx=None,
        dropout_p=0.2,
        original_weights=None  # 기존 학습된 가중치 전달
    ):
        super(PrunedSECell_OriginalWeight, self).__init__()
        
        self.drop_path_prob = drop_path_prob
        self.reduction = reduction
        self.cell_idx = cell_idx
        
        # 가지치기 정보 처리 (이 셀에 해당하는 노드들만)
        self.pruned_nodes_local = set()
        if pruned_nodes_info:
            for (c_idx, n_idx) in pruned_nodes_info.keys():
                if c_idx == self.cell_idx:
                    self.pruned_nodes_local.add(n_idx)
        
        # Preprocessing layers (기존과 동일)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)
        
        # Genotype 정보 추출
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        
        self._compile(C, op_names, indices, concat, reduction, affine)
        
        # 스칼라 가중치 (기존 학습된 값으로 고정)
        if not reduction:
            if original_weights is not None:
                # 기존 학습된 가중치로 고정
                self.register_buffer('node_weights', original_weights.clone())
            else:
                # fallback: 1로 고정
                self.register_buffer('node_weights', torch.ones(len(op_names) // 2))
        else:
            self.node_weights = None
        
        # SE Attention 모듈들 (가지치기된 노드에만)
        self.se_modules = nn.ModuleDict()
        for node_idx in self.pruned_nodes_local:
            self.se_modules[str(node_idx)] = SEAttentionWeightedAggregation(
                channels=C,
                num_inputs=2,
                reduction_ratio=4,
                dropout_p=dropout_p
            )
    
    def _compile(self, C, op_names, indices, concat, reduction, affine):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, affine=affine)
            self._ops.append(op)
        self._indices = indices
    
    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        for i in range(self._steps):
            h1_input = states[self._indices[2 * i]]
            h2_input = states[self._indices[2 * i + 1]]
            
            if i in self.pruned_nodes_local:
                # 가지치기된 노드: operation 삭제, SE로 직접 처리
                s = self.se_modules[str(i)]([h1_input, h2_input])
            else:
                # 기존 노드: operation 수행 + 기존 학습된 스칼라 가중치 (고정)
                op1 = self._ops[2 * i]
                op2 = self._ops[2 * i + 1]
                h1 = op1(h1_input)
                h2 = op2(h2_input)
                
                if self.training and drop_prob > 0.:
                    if not isinstance(op1, nn.Identity):
                        h1 = drop_path(h1, drop_prob, self.training)
                    if not isinstance(op2, nn.Identity):
                        h2 = drop_path(h2, drop_prob, self.training)
                
                s = h1 + h2
                if not self.reduction:
                    # 기존 학습된 가중치로 고정
                    s = self.node_weights[i] * s
            
            states.append(s)
        
        return torch.cat([states[i] for i in self._concat], dim=1)
    
    def arch_parameters(self):
        # SE 모듈의 파라미터들만 반환 (스칼라 가중치는 고정)
        params = []
        for se_module in self.se_modules.values():
            params.extend(list(se_module.parameters()))
        return params


class PrunedSECell_Learnable(nn.Module):
    """
    가지치기된 셀 - 스칼라 파라미터 학습 버전
    - 가지치기된 노드: operation 삭제 + SE Attention으로 교체
    - 기존 노드: 스칼라 가중치도 함께 학습
    - SE 모듈 + 스칼라 가중치 모두 fine-tuning에 참여
    """
    def __init__(
        self, genotype,
        C_prev_prev, C_prev, C,
        reduction, reduction_prev,
        affine=True, drop_path_prob=0.0,
        pruned_nodes_info=None,
        cell_idx=None,
        dropout_p=0.2
    ):
        super(PrunedSECell_Learnable, self).__init__()
        
        self.drop_path_prob = drop_path_prob
        self.reduction = reduction
        self.cell_idx = cell_idx
        
        # 가지치기 정보 처리 (이 셀에 해당하는 노드들만)
        self.pruned_nodes_local = set()
        if pruned_nodes_info:
            for (c_idx, n_idx) in pruned_nodes_info.keys():
                if c_idx == self.cell_idx:
                    self.pruned_nodes_local.add(n_idx)
        
        # Preprocessing layers (기존과 동일)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=affine)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=affine)
        
        # Genotype 정보 추출
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        
        self._compile(C, op_names, indices, concat, reduction, affine)
        
        # 스칼라 가중치 (학습 가능)
        if not reduction:
            self.node_weights = nn.Parameter(torch.ones(len(op_names) // 2))
        else:
            self.node_weights = None
        
        # SE Attention 모듈들 (가지치기된 노드에만)
        self.se_modules = nn.ModuleDict()
        for node_idx in self.pruned_nodes_local:
            self.se_modules[str(node_idx)] = SEAttentionWeightedAggregation(
                channels=C,
                num_inputs=2,
                reduction_ratio=4,
                dropout_p=dropout_p
            )
    
    def _compile(self, C, op_names, indices, concat, reduction, affine):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, affine=affine)
            self._ops.append(op)
        self._indices = indices
    
    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        for i in range(self._steps):
            h1_input = states[self._indices[2 * i]]
            h2_input = states[self._indices[2 * i + 1]]
            
            if i in self.pruned_nodes_local:
                # 가지치기된 노드: operation 삭제, SE로 직접 처리
                s = self.se_modules[str(i)]([h1_input, h2_input])
            else:
                # 기존 노드: operation 수행 + 학습 가능한 스칼라 가중치
                op1 = self._ops[2 * i]
                op2 = self._ops[2 * i + 1]
                h1 = op1(h1_input)
                h2 = op2(h2_input)
                
                if self.training and drop_prob > 0.:
                    if not isinstance(op1, nn.Identity):
                        h1 = drop_path(h1, drop_prob, self.training)
                    if not isinstance(op2, nn.Identity):
                        h2 = drop_path(h2, drop_prob, self.training)
                
                s = h1 + h2
                if not self.reduction:
                    # 학습 가능한 가중치 적용
                    s = self.node_weights[i] * s
            
            states.append(s)
        
        return torch.cat([states[i] for i in self._concat], dim=1)
    
    def arch_parameters(self):
        # SE 모듈 + 스칼라 가중치 모두 반환
        params = []
        for se_module in self.se_modules.values():
            params.extend(list(se_module.parameters()))
        if self.node_weights is not None:
            params.append(self.node_weights)
        return params


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(C, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.mean([2, 3])  # Global Average Pooling
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


    
# NetworkCIFAR 클래스 정의 수정
class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype,
                 drop_path_prob=0.0, affine=True, cell_type='search',
                 **kwargs):
        super().__init__()
        self.drop_path_prob = drop_path_prob
        self._auxiliary = auxiliary
        self.layers = layers
        self.cell_type = cell_type

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            nn.ReLU(inplace=True)
        )

        C_prev_prev, C_prev = C_curr, C_curr
        self.cells = nn.ModuleList()
        reduction_layers = [layers // 3, 2 * layers // 3]
        multiplier = len(genotype.normal_concat)

        # cell_type에 따라 사용할 Cell 클래스 선택
        if cell_type == 'search':
            cell_cls = SearchCell
        elif cell_type == 'train':
            cell_cls = TrainCell
        elif cell_type == 'normal_weighted':
            cell_cls = NormalOnlyWeightedSearchCell
        elif cell_type == "branch_search":
            cell_cls = WeightedBranchCell
        elif cell_type == "branch_normal_search":
            cell_cls = WeightedBranchNormalCell
        elif cell_type == "LearnableFusion_train":
            cell_cls = LearnableFusionCell
        elif cell_type == 'traink1':
            cell_cls = TrainK1Cell
        elif cell_type == "SEFusion_train":
            cell_cls = SEFusionCell
        # ================ 새로 추가된 cell_type들 ================
        elif cell_type == "pruned_se_fixed":
            cell_cls = PrunedSECell_Fixed
        elif cell_type == "pruned_se_learnable":
            cell_cls = PrunedSECell_Learnable
        elif cell_type == "pruned_se_original":
            cell_cls = PrunedSECell_OriginalWeight
        # =========================================================
        else:
            raise ValueError(f"Invalid cell_type: {cell_type}")

        for i in range(layers):
            reduction = i in reduction_layers
            if reduction:
                C_curr *= 2
            reduction_prev = i > 0 and (i - 1) in reduction_layers

            cell_args = {
                'genotype': genotype, 'C_prev_prev': C_prev_prev,
                'C_prev': C_prev, 'C': C_curr, 'reduction': reduction,
                'reduction_prev': reduction_prev, 'affine': affine,
                'drop_path_prob': drop_path_prob
            }

            # 특정 cell_type들에 추가 인자 전달
            if cell_type in ["LearnableFusion_train", "SEFusion_train", "pruned_se_fixed", "pruned_se_learnable", "pruned_se_original"]:
                if 'pruned_nodes_info' in kwargs:
                    cell_args['pruned_nodes_info'] = kwargs['pruned_nodes_info']
                if 'dropout_p' in kwargs:
                    cell_args['dropout_p'] = kwargs['dropout_p']
                if 'original_weights_dict' in kwargs and cell_type == "pruned_se_original":
                    # 원본 가중치 전달 (셀별로)
                    cell_key = f'cells.{i}'
                    if cell_key in kwargs['original_weights_dict']:
                        cell_args['original_weights'] = kwargs['original_weights_dict'][cell_key]
                cell_args['cell_idx'] = i

            cell = cell_cls(**cell_args)
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            drop_prob = self.drop_path_prob * i / (self.layers - 1) if self.drop_path_prob > 0. else 0.0
            s0, s1 = s1, cell(s0, s1, drop_prob)
            if i == 2 * self.layers // 3 and self._auxiliary and self.training:
                logits_aux = self.auxiliary_head(s1)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux