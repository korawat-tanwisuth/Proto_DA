import random
import time
import warnings
import sys
import argparse
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

sys.path.append('.')
from dalib.adaptation.proto import ProtoLoss, ImageClassifier
import dalib.vision.datasets as datasets
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator
from tools.transforms import ResizeImage
from tools.lr_scheduler import StepwiseLR
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    
    val_transform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    train_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=train_transform, subsample=args.sub_s)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, task=args.target, download=True, transform=train_transform, subsample=args.sub_t)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.bs_tgt,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_transform, subsample=args.sub_t)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, evaluate=True, download=True, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    classifier = ImageClassifier(backbone, train_source_dataset.num_classes).to(device)

    # define loss function
    num_classes = train_source_dataset.num_classes
    domain_loss = ProtoLoss(args.nav_t, args.beta, num_classes, device, args.s_par).to(device)
    domain_loss.true_prop = torch.Tensor(train_target_dataset.proportion).unsqueeze(1).to(device)
    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=0.75)
    beta_scheduler = StepwiseLR(None, init_lr=args.beta, gamma=args.lr_gamma, decay_rate=0.75)
 
    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_loss, optimizer,
              lr_scheduler, beta_scheduler, epoch, args)
        print(domain_loss.prop.squeeze(1).tolist())
        # evaluate on validation set
        acc1 = validate(val_loader, classifier, domain_loss, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(best_model)
    acc1 = validate(test_loader, classifier, domain_loss, args)
    print("test_acc1 = {:3.1f}".format(acc1))

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, domain_loss: ProtoLoss, optimizer: SGD,
          lr_scheduler: StepwiseLR, beta_scheduler: StepwiseLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    transfer_losses = AverageMeter('Transfer Loss', ':6.2f')
    prop_losses = AverageMeter('Prop Loss', ':6.6f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, transfer_losses, prop_losses, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_loss.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        lr_scheduler.step()
        beta_scheduler.step()

        # measure data loading time
        data_time.update(time.time() - end)
           
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        x_list = [x_s, x_t]
      
        combined_x = torch.cat(x_list, dim=0)
        y, f = model(combined_x)
        del x_list

        prototypes_s = model.head.weight.data.clone()

        f_chunks = torch.split(f, [args.batch_size] + [args.bs_tgt], dim=0) 
        f_s, f_t = f_chunks[0], f_chunks[-1]
        y_chunks = torch.split(y, [args.batch_size] + [args.bs_tgt], dim=0)
        y_s, y_t = y_chunks[0], y_chunks[-1]
 
        cls_loss = F.cross_entropy(y_s, labels_s)
    
        domain_loss.beta = beta_scheduler.get_lr()
        transfer_loss = domain_loss(prototypes_s, f_t)
        loss = cls_loss + transfer_loss * args.trade_off 
        prop_loss = torch.abs(domain_loss.true_prop - domain_loss.prop).mean()
        
        cls_acc = accuracy(y_s, labels_s)[0]

        prop_losses.update(prop_loss.item(), prototypes_s.size(0)) 
        transfer_losses.update(transfer_loss.item(), y_s.size(0)) 
        losses.update(loss.item(), y_s.size(0))
        cls_accs.update(cls_acc.item(), y_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, domain_loss: nn.Module, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            y_t, f_t = model(images)
            loss = F.cross_entropy(y_t, target)

            mu_s = model.head.weight.data.clone()
            sim_mat = torch.matmul(mu_s, f_t.T)
            output = domain_loss.get_pos_logits(sim_mat, domain_loss.prop).T
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size for source (default: 32)')
    parser.add_argument('--bs_tgt', default=96, type=int, 
                        help='target batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-nav_t', '--nav_t', default=1, type=float,
                        help='temperature for the navigator')
    parser.add_argument('-beta', '--beta', default=0, type=float,
            help='momentum coefficient')
    parser.add_argument('--s_par', default=0.5, type=float, 
                        help='s_par')
    parser.add_argument('--sub_s', default=False, action='store_true')
    parser.add_argument('--sub_t', default=False, action='store_true')



    args = parser.parse_args()
    print(args)
    main(args)

