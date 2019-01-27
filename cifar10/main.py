'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
import models
import importlib
import logging
import re
from datetime import datetime
import collections
import utils
from functools import partial
def partial(func, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*(args + fargs), **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--net-type', default='CifarResNetBasic', type=str,
                    help='the type of net (CifarResNetNoBNBasic, CifarPlainBNConvReLUNet, CifarResNetBasic,'
                         ' CifarPlainNetBasic or CifarPlainNetBasicNoBatchNorm or ResNetBasic or ResNetBottleneck)')
parser.add_argument('--num-blocks', default='3-3-3', type=str, help='starting net')
parser.add_argument('--norm-init', action='store_true', help='use norm initialization')
parser.add_argument('--adaptive-decay', default=None, type=float, help='coefficient to adapt weight decay to Gaussian prior')

parser.add_argument('--init-batch-size', default=2000, type=int, help='batch size')
parser.add_argument('--loss-scaler', default=1.0, type=float, help='The factor to scale loss')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--backnorm', action='store_true', help='use backnorm')
parser.add_argument('--norm-layers', default='torch.nn.Conv2d', type=str, help='the type of layers whose inputs are back normalized. Connect multiple types by +')
parser.add_argument('--reinit-std', default=None, type=float, help='reinitialization std')
parser.add_argument('--norm-dim', default=None, type=int, help='the dim to add backnorm')
parser.add_argument('--epochs', default=200, type=int, help='the number of epochs')
parser.add_argument('--print-freq', default=3910, type=int, help='the frequency to print')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

net_arch = list(map(int, args.num_blocks.split('-')))
save_path = os.path.join('./results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    raise OSError('Directory {%s} exists. Use a new one.' % save_path)
logging.basicConfig(filename=os.path.join(save_path, 'log.txt'), level=logging.INFO)
logger = logging.getLogger('main')
logger.addHandler(logging.StreamHandler())
logger.info("Saving to %s", save_path)
logger.info("Running arguments: %s", args)
best_acc = 0  # best test accuracy
trained_batchs = 0

def remove_hooks(hs):
    for h in hs:
        h.remove()

def add_backward_hooks(model):

    def myhook(m, grad_input, grad_output, name=None):
        global trained_batchs
        input_idx = 0
        if isinstance(m, nn.Linear):
            input_idx = 1
        if grad_input[input_idx] is None:
            return grad_input
        if args.norm_dim is None:
            grad_std = grad_input[input_idx].std() + 1.0e-10
            grad_mean = grad_input[input_idx].mean()
        else:
            grad_std = grad_input[input_idx].std(dim=args.norm_dim, keepdim=True) + 1.0e-10
            grad_mean = grad_input[input_idx].mean(dim=args.norm_dim, keepdim=True)

        if trained_batchs % args.print_freq == 0:
            logger.info('%s: mean (%.8f), std (%.8f)' % (
                name, grad_mean.abs().mean(), grad_std.mean()))

        norm_grad_input = (grad_input[input_idx] - grad_mean) / grad_std + grad_mean
        # return (norm_grad_input, grad_input[1], grad_input[2])
        # return (grad_input[0], grad_input[1], grad_input[2])
        grad_tuple = tuple(list(grad_input[0:input_idx]) + [norm_grad_input] + list(grad_input[input_idx+1:]))
        return grad_tuple

    handles = []
    for norm_layer in args.norm_layers.split('+'):
        path, layer_name = os.path.splitext(norm_layer)
        layer_name = layer_name[1:]
        logger.info('Hooking layer %s.%s' % (path, layer_name))
        modname = importlib.import_module(path)
        for idx, m in enumerate(model.named_modules()):
            if isinstance(m[1], getattr(modname, layer_name)) and (not re.match('.*shortcut.*', m[0])):
                logger.info('\t{} registering backward hook...'.format(m[0]))
                h = m[1].register_backward_hook(hook=partial(myhook, name=m[0]))
                handles.append(h)
    return handles

activation_mean_std = collections.OrderedDict()
hooked_layers = collections.OrderedDict()
def add_forward_hooks(model):

    def myhook(m, input, output, name=None):
        activation_mean_std[name] = {'mean': output.mean(), 'std': output.std()}
        if trained_batchs % args.print_freq == 1:
            logger.info('%s: mean (%.8f), std (%.8f)' % (
                name, activation_mean_std[name]['mean'], activation_mean_std[name]['std']))

    handles = []
    logger.info('Hooking Conv2d ...')
    for idx, m in enumerate(model.named_modules()):
        if isinstance(m[1], nn.Conv2d):
            logger.info('\t{} registering forward hook...'.format(m[0]))
            h = m[1].register_forward_hook(hook=partial(myhook, name=m[0]))
            handles.append(h)
            hooked_layers[m[0]] = m[1]
    return handles

def decay_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1

# Training
def norm_initialization(max_iter, net, layer_name, layer, loader, criterion, threshold=1.0e-2, device='cuda'):
    logger.info('\nReinitializing %s for maximum %d iterations' % (layer_name, max_iter))
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        orig_std = layer.weight.data.std()
        layer.weight.data.normal_(0, orig_std)
        if layer.bias is not None:
            layer.bias.data.normal_(0, orig_std)
        min_std = 0.1 * orig_std
        max_std = 10.0 * orig_std
        current_std = orig_std
        logger.info('\tinitial stds: [%.6f, %.6f, %.6f]' % (min_std, current_std, max_std))
        found = False
        ema_mean = utils.ExponentialMovingAverage(decay=0.0, scale=True)
        ema_std = utils.ExponentialMovingAverage(decay=0.0, scale=True)
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            break
        for iter in range(max_iter):
            # for batch_idx, (inputs, targets) in enumerate(loader):
                outputs = net(inputs)
                loss = criterion(outputs, targets) * args.loss_scaler

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # adjust std
                ema_mean.push(activation_mean_std[layer_name]['mean'])
                ema_std.push(activation_mean_std[layer_name]['std'])
                if 0 == iter % 5 or iter == max_iter - 1:
                    logger.info('\t %d iteration ==> mean: %.6f, std: %.6f | avg mean %.6f, avg std %.6f'
                                % (iter,
                                   activation_mean_std[layer_name]['mean'],
                                   activation_mean_std[layer_name]['std'],
                                   ema_mean.average(),
                                   ema_std.average()))

                if abs(ema_std.average() - 1.0) < threshold:
                    found = True
                    break

                if ema_std.average() > 1.0:
                    max_std = min(current_std * 1.5, max_std - 1.1e-4)
                else:
                    min_std = max(current_std * 0.5, min_std + 1.1e-4)
                current_std = (min_std + max_std) / 2.0
                # logger.info('\tsetting current stds: [%.6f, %.6f, %.6f]' %(min_std, current_std, max_std))
                layer.weight.data.normal_(0, current_std)
                if layer.bias is not None:
                    layer.bias.data.normal_(0, current_std)

                if max_std - min_std < 1.0e-4:
                    logger.info('\tfinised because of a small search range [%.6f, %.6f]' % (min_std, max_std))
                    break


        logger.info('\tfound=%r at iter %d. mean: %.6f, std: %.6f | avg mean %.6f, avg std %.6f'
                    % (found, iter,
                       activation_mean_std[layer_name]['mean'],
                       activation_mean_std[layer_name]['std'],
                       ema_mean.average(),
                       ema_std.average()))
        logger.info('\tOriginal std: %.6f -> new std: %.6f' % (orig_std, current_std))

    return found, current_std

# Training
def train(epoch, net, loader, optimizer, criterion, device='cuda'):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global trained_batchs
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets) * args.loss_scaler
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if 0 == batch_idx % 100 or batch_idx == len(loader) - 1:
            logger.info('(%d/%d) ==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (
                        batch_idx + 1, len(loader), train_loss / (batch_idx + 1), 100. * correct / total, correct,
                        total))
        trained_batchs += 1
    return train_loss / len(loader), 100. * correct / total

def test(epoch, net, loader, criterion, device='cuda'):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets) * args.loss_scaler

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if 0 == batch_idx % 100 or batch_idx == len(loader) - 1:
                logger.info('(%d/%d) ==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx+1, len(loader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        logger.info('Saving the best model %.3f @ %d ...' % (acc, epoch))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

    return test_loss / len(loader), acc

def plot_curves(curves, save_path):
    # plotting
    clr1 = (0.5, 0., 0.)
    clr2 = (0.0, 0.5, 0.)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss', color=clr1)
    ax1.tick_params(axis='y', colors=clr1)
    ax2.set_ylabel('Accuracy (%)', color=clr2)
    ax2.tick_params(axis='y', colors=clr2)

    markersize = 12
    ax1.semilogy(curves[:, 0], curves[:, 1], '--', color=clr1,
                 markersize=markersize)
    ax1.semilogy(curves[:, 0], curves[:, 3], '-', color=clr1,
                 markersize=markersize)
    ax2.plot(curves[:, 0], curves[:, 2], '--', color=clr2,
             markersize=markersize)
    ax2.plot(curves[:, 0], curves[:, 4], '-', color=clr2,
             markersize=markersize)

    # ax2.set_ylim(bottom=40, top=100)
    ax1.legend(('Train loss', 'Val loss'), loc='lower right')
    ax2.legend(('Train accuracy', 'Val accuracy', 'Val moving avg'), loc='lower left')
    fig.savefig(os.path.join(save_path, 'curves-vs-epochs.pdf'))

def main():
    device = 'cuda' # if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    logger.info('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_init_loader = torch.utils.data.DataLoader(trainset, batch_size=args.init_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Model
    logger.info('==> Building model..')
    net = getattr(models, args.net_type)(net_arch, args.reinit_std)
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = CifarResNetBasic([1,1,1])
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    if args.backnorm:
        add_backward_hooks(net)

    criterion = nn.CrossEntropyLoss()
    net = net.to(device)

    params_decay_options = {}
    if args.norm_init:
        fwd_hks = add_forward_hooks(net)
        for layer_name in hooked_layers:
            found, current_std = norm_initialization(250, net, layer_name, hooked_layers[layer_name], train_init_loader, criterion)
            if args.adaptive_decay is not None:
                ad_wd = args.adaptive_decay * 0.5 / (current_std ** 2) * args.batch_size / len(trainloader) * args.loss_scaler
                params_decay_options[id(hooked_layers[layer_name].weight)] = \
                    {'params': [hooked_layers[layer_name].weight],
                     'weight_decay': ad_wd}
                if hooked_layers[layer_name].bias is not None:
                    params_decay_options[id(hooked_layers[layer_name].bias)] = \
                        {'params': [hooked_layers[layer_name].bias],
                         'weight_decay': ad_wd}
        remove_hooks(fwd_hks)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        logger.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    params_options = []
    for name, param in net.named_parameters():
        if id(param) in params_decay_options:
            logger.info('%s weight decay is adapted to %.6f' % (name, params_decay_options[id(param)]['weight_decay']))
            params_options.append(params_decay_options[id(param)])
        else:
            logger.info('%s weight decay is using the default' % (name))
            params_options.append({'params': [param]})
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=(5e-4)*args.loss_scaler)
    optimizer = optim.SGD(params_options, lr=args.lr, momentum=0.9, weight_decay=(5e-4) * args.loss_scaler)



    curves = np.zeros((args.epochs, 5))
    for epoch in range(start_epoch, start_epoch+args.epochs):
        if epoch == args.epochs / 2 or epoch == args.epochs * 3 / 4:
            logger.info('======> decaying learning rate')
            decay_learning_rate(optimizer)

        curves[epoch, 0] = epoch
        curves[epoch, 1], curves[epoch, 2] = train(epoch, net, trainloader, optimizer, criterion)
        curves[epoch, 3], curves[epoch, 4] = test(epoch, net, testloader, criterion)
        if epoch % 5 == 0:
            plot_curves(curves[:epoch+1, :], save_path)

    plot_curves(curves, save_path)
    logger.info('curves: \n {}'.format(np.array_str(curves)))
    np.savetxt(os.path.join(save_path, 'curves.dat'), curves)



if __name__ == '__main__':
    main()
    logger.info('Done!')