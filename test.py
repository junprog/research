import torch
import time
import os
import sys
import math

import numpy as np
from matplotlib import pyplot as plt
from utils import AverageMeter

def test(data_loader, model, logger, device, opts):

    model.eval()

    MAE_losses = AverageMeter()
    RMSE_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    with torch.no_grad():
        for i, (inputs, target, num) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            inputs = inputs.to(device)
            #num = torch.tensor(num, dtype=torch.float32)
            #num = num.cuda()

            outputs = model(inputs)

            output_sum = torch.sum(outputs)

            MAE = torch.abs(torch.sub(output_sum, num.item())) 
            RMSE_tmp = torch.pow(torch.sub(output_sum, num.item()),2)

            MAE_losses.update(MAE.cpu().item(), inputs.size(0))
            RMSE_losses.update(RMSE_tmp.cpu().item(), inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('iterate: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'MAE {mae.val:.4f} ({mae.avg:.4f})\t'
                'RMSE_notpow {rmse.val:.4f} ({rmse.avg:.4f})\t'.format(
                    i+1,
                    len(data_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    mae=MAE_losses,
                    rmse=RMSE_losses)
            )
            
            if i % 10 == 0:
                numpy_in_1 = inputs[0,:,:,:].to('cpu').clone().numpy().copy()
                numpy_in_1 = numpy_in_1.transpose(1,2,0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                numpy_in_1 = numpy_in_1*std+mean

                numpy_out = outputs[:,:,:,:].to('cpu').clone().numpy().copy().squeeze()
                #numpy_target = target[:,:,:,:].to('cpu').detach().numpy().copy().squeeze()
                numpy_target = target[:,:,:,:].clone().numpy().copy().squeeze()


                fig = plt.figure()
                a_1 = fig.add_subplot(2,1,1)
                a_2 = fig.add_subplot(2,2,3)
                a_3 = fig.add_subplot(2,2,4)

                y, _ = numpy_target.shape

                a_1.imshow(numpy_in_1)
                a_1.set_title('Input Image')
                a_2.imshow(numpy_target, cmap='jet')
                a_2.annotate('{}'.format(int(num)), xy=(10, y-10), fontsize=16, color='white')
                a_2.set_title('Ground Truth')
                a_3.imshow(numpy_out, cmap='jet')
                a_3.annotate('{:.3f}'.format(output_sum.item()), xy=(10, y-10), fontsize=16, color='white')
                a_3.set_title('Prediction')

                plt.tight_layout()
                plt.savefig(os.path.join(opts.results_path, 'images', 'shanghaitech_partB_test_{}.png'.format(i)))
                plt.close(fig)

            #del inputs, target
    
    logger.log({
        'MAE': MAE_losses.avg,
        'RMSE': math.sqrt(RMSE_losses.avg)
    })