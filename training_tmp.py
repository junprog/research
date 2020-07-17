import torch
import time
import os
import sys
from matplotlib import pyplot as plt

from utils import AverageMeter

def train_epoch(epoch, data_loader, model, criterion):

    model.eval()

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        inputs = inputs.cuda()
        target = target.cuda()

        outputs = model(inputs)

        numpy_out = outputs[0,:,:,:].to('cpu').detach().numpy().copy().squeeze()
        numpy_target = target[0,:,:,:].to('cpu').detach().numpy().copy().squeeze()

        fig = plt.figure()
        a = fig.add_subplot(1,2,1)
        b = fig.add_subplot(1,2,2)
        a.imshow(numpy_target)
        b.imshow(numpy_out)
        plt.show()

        loss = criterion(outputs, target)

        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                  epoch,
                  i+1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses)
        )