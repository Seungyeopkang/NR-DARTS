import torch
import torch.nn as nn
import sys
from operations import *
from utils import drop_path
from genotypes import Genotype


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
            self.node_weights = None

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
            if not self.reduction:
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


class LearnableFusionCell(nn.Module):
    def __init__(
        self, genotype, 
        C_prev_prev, C_prev, C, 
        reduction, reduction_prev, 
        affine=True, drop_path_prob=0.0,
        
        pruned_nodes_info=None,
        cell_idx=None

    ):
        super().__init__()
        self.drop_path_prob = drop_path_prob
        self.cell_idx = cell_idx


        self.pruned_nodes_local_info = {}
        if pruned_nodes_info:
            for c_idx, n_idx in pruned_nodes_info.keys():
                if c_idx == self.cell_idx:
                    self.pruned_nodes_local_info[n_idx] = True


        self.learnable_aggregators = nn.ModuleDict()
        for i in range(len(genotype.normal) // 2 if not reduction else len(genotype.reduce) // 2): 
            if i in self.pruned_nodes_local_info:
                self.learnable_aggregators[str(i)] = LearnableWeightedAggregation(C=C, num_inputs=2)

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
            self._ops += [op]

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
                
                s = h1 + h2 # 기존 방식: 연산 결과 합산
            
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

        self.learnable_aggregators = nn.ModuleDict()
        num_intermediate_nodes = len(genotype.normal) // 2 if not reduction else len(genotype.reduce) // 2
        for i in range(num_intermediate_nodes):
            if i in self.pruned_nodes_local_info:

                self.learnable_aggregators[str(i)] = SEAttentionWeightedAggregation(
                    num_inputs=2,
                    channels=C,
                    reduction_ratio=4
                )


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


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(C, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.mean([2, 3])  # Global Average Pooling
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


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

        if cell_type == 'train':
            cell_cls = TrainCell
        elif cell_type == 'normal_weighted':
            cell_cls = NormalOnlyWeightedSearchCell
        elif cell_type == "LearnableFusion_train":
            cell_cls = LearnableFusionCell
        elif cell_type == "SEFusion_train":
            cell_cls = SEFusionCell
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

            if cell_type in ["LearnableFusion_train", "SEFusion_train"]:
                if 'pruned_nodes_info' in kwargs:
                    cell_args['pruned_nodes_info'] = kwargs['pruned_nodes_info']
                if 'dropout_p' in kwargs:
                    cell_args['dropout_p'] = kwargs['dropout_p']
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