import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import NetworkCIFAR
from models.utils import AverageMeter, accuracy, Cutout, _data_transforms_cifar10
from models.genotypes import DARTS


parser = argparse.ArgumentParser("Node Importance Search")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=120, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--save', type=str, default='./exp/search', help='experiment save directory')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, total_epochs, args):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{total_epochs}] Training", leave=False)
    for step, (input, target) in enumerate(pbar):
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()
        logits, _ = model(input)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        prec1, _ = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)

        pbar.set_postfix(loss=losses.avg, acc=f'{top1.avg:.2f}%')
    
    return losses.avg, top1.avg

def infer(data_loader, model, criterion, device, desc="Validation"):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    pbar = tqdm(data_loader, desc=f"[{desc}]", leave=False)
    with torch.no_grad():
        for input, target in pbar:
            input, target = input.to(device), target.to(device)
            logits, _ = model(input)
            loss = criterion(logits, target)
            
            prec1, _ = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            pbar.set_postfix(loss=losses.avg, acc=f'{top1.avg:.2f}%')

    return losses.avg, top1.avg

def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    os.makedirs(args.save, exist_ok=True)
    best_model_path = os.path.join(args.save, f"search-seed{args.seed}-best.pth")
    log_path = os.path.join(args.save, f'search-seed{args.seed}-log.json')


    train_transform, valid_transform = _data_transforms_cifar10(args)
    full_train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)


    train_len = int(0.75 * len(full_train_data))
    valid_len = len(full_train_data) - train_len
    train_data, valid_data = random_split(full_train_data, [train_len, valid_len])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)


    criterion = nn.CrossEntropyLoss().to(device)
    model = NetworkCIFAR(
        args.init_channels, 10, args.layers, args.auxiliary,
        DARTS, args.drop_path_prob,
        cell_type="normal_weighted"
    ).to(device)
    

    normal_params = []
    for cell in model.cells:
        if not cell.reduction:
            normal_params.extend(list(cell.parameters()))

    optimizer = optim.SGD(
        normal_params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=float(args.epochs))

    train_losses, valid_losses, valid_accuracies = [], [], []
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n--- Epoch {epoch+1}/{args.epochs} | Learning Rate: {current_lr:.6f} ---")
        
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, args.epochs, args)
        train_losses.append(train_loss)
        print(f"  -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        scheduler.step()

        valid_loss, valid_acc = infer(valid_loader, model, criterion, device, desc="Validation")
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        print(f"  -> Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"   New best model saved with validation accuracy: {best_acc:.2f}%")

    training_time = time.time() - start_time
    print(f"\n{'='*40}")
    print("Training phase finished!")
    print(f"Total training time: {training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Best model saved at: {best_model_path}")
    print(f"{'='*40}")

    print(f"\nLoading best model from {best_model_path} for testing...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc = infer(test_loader, model, criterion, device, desc="Testing")
        print(f"[Final Test Result]  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    else:
        print("Could not find the best model to test. Skipping test phase.")
        test_acc = 0.0

    log_data = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_accuracies': valid_accuracies,
        'test_accuracy': test_acc,
        'best_valid_accuracy': best_acc,
        'training_time_seconds': training_time
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"Log file saved to: {log_path}")
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)