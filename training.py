import torch
import time
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from utils import AverageMeter

def train_epoch(epoch, data_loader, model, criterion, optimizer, epoch_logger, opts, scheduler=None):
    if scheduler is not None:
        scheduler.step()

    if epoch < 40:
        for params in model.feature_extracter.parameters():
            params.requires_grad = False
    else:
        for params in model.feature_extracter.parameters():
            params.requires_grad = True
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    for i, (inputs, target, _) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        inputs = inputs.cuda()
        target = target.cuda()

        outputs = model(inputs)

        loss = criterion(outputs, target)

        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opts.checkpoint == 0:
        save_file_path = os.path.join(opts.results_path, 'saved_model', 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

    numpy_in_1 = inputs[0,:,:,:].to('cpu').detach().numpy().copy()
    numpy_in_1 = numpy_in_1.transpose(1,2,0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    numpy_in_1 = numpy_in_1*std+mean

    numpy_in_2 = inputs[1,:,:,:].to('cpu').detach().numpy().copy()
    numpy_in_2 = numpy_in_2.transpose(1,2,0)
    numpy_in_2 = numpy_in_2*std+mean

    numpy_in_3 = inputs[2,:,:,:].to('cpu').detach().numpy().copy()
    numpy_in_3 = numpy_in_3.transpose(1,2,0)
    numpy_in_3 = numpy_in_3*std+mean

    numpy_in_4 = inputs[3,:,:,:].to('cpu').detach().numpy().copy()
    numpy_in_4 = numpy_in_4.transpose(1,2,0)
    numpy_in_4 = numpy_in_4*std+mean


    numpy_out = outputs[0:4,:,:,:].to('cpu').detach().numpy().copy().squeeze()
    numpy_target = target[0:4,:,:,:].to('cpu').detach().numpy().copy().squeeze()

    fig = plt.figure()
    a_1 = fig.add_subplot(3,4,1)
    a_2 = fig.add_subplot(3,4,5)
    a_3 = fig.add_subplot(3,4,9)

    b_1 = fig.add_subplot(3,4,2)
    b_2 = fig.add_subplot(3,4,6)
    b_3 = fig.add_subplot(3,4,10)

    c_1 = fig.add_subplot(3,4,3)
    c_2 = fig.add_subplot(3,4,7)
    c_3 = fig.add_subplot(3,4,11)

    d_1 = fig.add_subplot(3,4,4)
    d_2 = fig.add_subplot(3,4,8)
    d_3 = fig.add_subplot(3,4,12)

    a_1.imshow(numpy_in_1)
    a_2.imshow(numpy_target[0], cmap='jet')
    a_3.imshow(numpy_out[0], cmap='jet')

    b_1.imshow(numpy_in_2)
    b_2.imshow(numpy_target[1], cmap='jet')
    b_3.imshow(numpy_out[1], cmap='jet')

    c_1.imshow(numpy_in_3)
    c_2.imshow(numpy_target[2], cmap='jet')
    c_3.imshow(numpy_out[2], cmap='jet')

    d_1.imshow(numpy_in_4)
    d_2.imshow(numpy_target[3], cmap='jet')
    d_3.imshow(numpy_out[3], cmap='jet')

    plt.savefig(os.path.join(opts.results_path, 'images', 'shanghaitech_partB_tr_fromMSE_{}.png'.format(epoch)))
    plt.close(fig)