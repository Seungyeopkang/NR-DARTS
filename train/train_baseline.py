import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import random_split
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import NetworkCIFAR
from models.utils import _data_transforms_cifar10
from models.genotypes import DARTS


parser = argparse.ArgumentParser("Baseline Training")
parser.add_argument('--save', type=str, default='./exp/baseline', help='experiment save directory')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def measure_flops(model, input_size=(1, 3, 32, 32), device='cuda'):
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)


    def get_tensor_numel(tensor_value):
        try:
            tensor_type = tensor_value.type()
            if hasattr(tensor_type, 'sizes'):
                sizes = tensor_type.sizes()
                numel = 1
                for size in sizes:
                    numel *= size
                return numel
            else:
                return 1
        except:
            return 1


    def aten_softmax_flop_jit(inputs, outputs):
        numel = get_tensor_numel(outputs[0])
        return numel * 5


    def aten_mul_flop_jit(inputs, outputs):
        return get_tensor_numel(outputs[0])


    def aten_add_flop_jit(inputs, outputs):
        return get_tensor_numel(outputs[0])


    def aten_softplus_flop_jit(inputs, outputs):
        numel = get_tensor_numel(outputs[0])
        return numel * 4


    def aten_max_pool2d_flop_jit(inputs, outputs):
        numel = get_tensor_numel(outputs[0])
        kernel_flops = 4
        return numel * kernel_flops


    def aten_adaptive_avg_pool2d_flop_jit(inputs, outputs):
        numel = get_tensor_numel(inputs[0])
        return numel


    def aten_linear_flop_jit(inputs, outputs):
        try:
            input_numel = get_tensor_numel(inputs[0])
            weight_type = inputs[1].type()
            if hasattr(weight_type, 'sizes'):
                weight_sizes = weight_type.sizes()
                if len(weight_sizes) >= 2:
                    return input_numel * weight_sizes[0]
            return 0
        except:
            return 0


    def aten_relu_flop_jit(inputs, outputs):
        return get_tensor_numel(outputs[0])


    def aten_sigmoid_flop_jit(inputs, outputs):
        numel = get_tensor_numel(outputs[0])
        return numel * 4 


    def aten_view_flop_jit(inputs, outputs):
        return 0

    flops = FlopCountAnalysis(model, dummy_input)
    
    flops.set_op_handle("aten::softmax", aten_softmax_flop_jit)
    flops.set_op_handle("aten::mul", aten_mul_flop_jit)
    flops.set_op_handle("aten::add", aten_add_flop_jit)
    flops.set_op_handle("aten::softplus", aten_softplus_flop_jit)
    flops.set_op_handle("aten::max_pool2d", aten_max_pool2d_flop_jit)
    flops.set_op_handle("aten::adaptive_avg_pool2d", aten_adaptive_avg_pool2d_flop_jit)
    flops.set_op_handle("aten::linear", aten_linear_flop_jit)
    flops.set_op_handle("aten::addmm", aten_linear_flop_jit)
    flops.set_op_handle("aten::relu", aten_relu_flop_jit)
    flops.set_op_handle("aten::relu_", aten_relu_flop_jit)
    flops.set_op_handle("aten::sigmoid", aten_sigmoid_flop_jit)
    flops.set_op_handle("aten::view", aten_view_flop_jit)
    flops.set_op_handle("aten::reshape", aten_view_flop_jit)
    
    return flops.total() / 1e6


def measure_inference_latency(model, device, input_size=(1, 3, 32, 32), iterations=100, warmup=10):
    model.eval(); model.to(device)
    dummy_input = torch.randn(*input_size).to(device)
    
    with torch.no_grad():
        for _ in range(warmup): _ = model(dummy_input)
    
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for _ in range(iterations):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        return sum(timings) / len(timings)
    else:
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations): _ = model(dummy_input)
        end = time.time()
        return ((end - start) / iterations) * 1000


def measure_peak_memory(model, device, input_size=(1, 3, 32, 32)):
    if device.type != 'cuda': return 0.0
    model.eval(); model.to(device)
    torch.cuda.reset_peak_memory_stats(device=device)
    dummy_input = torch.randn(*input_size).to(device)
    with torch.no_grad(): _ = model(dummy_input)
    return torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)


def train_one_epoch(loader, model, criterion, optimizer, device, epoch_desc):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=epoch_desc)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        pbar.set_postfix(loss=f'{(total_loss/total):.3f}', acc=f'{(100.*correct/total):.2f}%')
    return total_loss / total, 100. * correct / total


def evaluate(loader, model, criterion, device, desc):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc)
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            pbar.set_postfix(loss=f'{(total_loss/total):.3f}', acc=f'{(100.*correct/total):.2f}%')
    return total_loss / total, 100. * correct / total


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    os.makedirs(args.save, exist_ok=True)

    train_transform, valid_transform = _data_transforms_cifar10(args)
    full_train = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    
    train_len = int(0.75 * len(full_train))
    val_len = len(full_train) - train_len
    train_data, val_data = random_split(full_train, [train_len, val_len])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model, Criterion, Optimizer
    model = NetworkCIFAR(args.init_channels, 10, args.layers, args.auxiliary, DARTS, args.drop_path_prob, cell_type="train").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_wts = None
    train_start = time.time()
    
    # Training Loop
    for epoch in range(args.epochs):
        epoch_desc_train = f"[Epoch {epoch+1}/{args.epochs}] Training"
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, device, epoch_desc_train)
        
        epoch_desc_val = f"[Epoch {epoch+1}/{args.epochs}] Validation"
        val_loss, val_acc = evaluate(val_loader, model, criterion, device, epoch_desc_val)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()
            print(f"Best model updated with validation accuracy: {val_acc:.2f}%")
        
        scheduler.step()
    
    training_time = time.time() - train_start
    
    # Test, Measure, and Save
    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(test_loader, model, criterion, device, "Testing Best Model")
    flops = measure_flops(model, device=device)
    latency = measure_inference_latency(model, device)
    max_memory_MB = measure_peak_memory(model, device)

    print("\n" + "="*50)
    print("           TRAINING FINISHED - RESULTS")
    print("="*50)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"FLOPs: {flops:.2f} M")
    print(f"Latency: {latency:.2f} ms")
    print(f"Max GPU Memory: {max_memory_MB:.2f} MB")
    print(f"Total Training Time: {training_time/3600:.2f} hours")
    print("="*50)
    
    history['test_acc'] = test_acc; history['flops'] = flops; history['training_time_sec'] = training_time
    history['inference_latency_ms'] = latency; history['memory_usage_MB'] = max_memory_MB

    torch.save(best_model_wts, os.path.join(args.save, f'baseline-seed{args.seed}-best.pth'))
    with open(os.path.join(args.save, f"baseline-seed{args.seed}-log.json"), 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Best model and logs saved in '{args.save}'")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)