import os
import shutil
import time
from collections import OrderedDict
from enum import Enum

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

best_acc1 = 0


def main():
    # load parameters
    m1 = torch.load('r12/checkpoint.pth.tar').to('cuda')
    m2 = torch.load('r22/checkpoint.pth.tar').to('cuda')
    m2o = torch.load('r2o/checkpoint.pth.tar').to('cuda')
    model1 = models.alexnet().to('cuda')
    model2 = models.alexnet().to('cuda')
    model1.load_state_dict(m1['state_dict'])
    model2.load_state_dict(m2['state_dict'])

    criterion = nn.CrossEntropyLoss()

    traindir = os.path.join('imagenet', 'train')
    valdir = os.path.join('imagenet', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=256, shuffle=False, num_workers=16,)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=16, )
    # validate(train_loader, model, criterion)
    # model1.to('cuda')
    # model2.to('cuda')
    # model_v.to('cuda')
    # model.to('cuda')
    val_model1 = validate(val_loader, model1, criterion)
    val_model2 = validate(val_loader, model2, criterion)
    # validate(val_loader, model_v, criterion)
    # validate(val_loader, model, criterion)

    train_avg_matching = []
    train_avg_vanilla = []
    val_avg_matching = []
    val_avg_vanilla = []
    for a in range(1, 21):
        matchedModel = OrderedDict()
        vanillaMatchedModel = OrderedDict()
        for k in m1['state_dict'].keys():
            matchedModel[k] = a * 0.05 * m1['state_dict'][k] + (1-a*0.05) * m2['state_dict'][k]
            vanillaMatchedModel[k] = a * 0.05 * m1['state_dict'][k] + (1-a*0.05) * m2o['state_dict'][k]
        modelMatched = models.alexnet()
        modelVanilla = models.alexnet()
        modelMatched.load_state_dict(matchedModel)
        modelVanilla.load_state_dict(vanillaMatchedModel)
        val_avg_matching.append(validate(val_loader, modelMatched, criterion))
        val_avg_vanilla.append(validate(val_loader, modelVanilla, criterion))

    # store results
    fname = 'results.txt'
    with open(fname, 'w') as f:
        f.write(f'Model 1: {val_model1}')
        f.write(f'Model 2: {val_model2}')
        f.write(f'Vanilla Interpolated Model: {val_avg_vanilla}')
        f.write(f'Matched Interpolated Model: {val_avg_matching}')


def validate(val_loader, model, criterion):
    model = model.to('cuda') if torch.cuda.is_available() else model

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.cuda(non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
