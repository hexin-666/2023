import os
import torch.nn as nn
import cv2
import torch
import ssim
from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
from torchvision.transforms import functional as Fs

def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    dataloader = train_dataloader(args.data_dir, args.data_dir1,args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img,input_img1, label_img1 = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            input_img1 = input_img1.to(device)
            label_img1 = label_img1.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img,input_img1)
            l3 = criterion(pred_img, label_img)

            l2 = ssim.ssim(pred_img, label_img1)

            loss_content = l3+0.10*(1-l2)


            label_fft3 = torch.fft.rfft(label_img, dim=2)
            pred_fft3 = torch.fft.rfft(pred_img, dim=2)
            label_fft2 = torch.fft.rfft(label_img1, dim=2)
            f3 = criterion(pred_fft3, label_fft3)
            f2 = criterion(pred_fft3, label_fft2)
            loss_fft = f3
            # + 0.1 * loss_fft
            loss = loss_content
            loss.backward()
            optimizer.step()
            # cv2.imwrite("1.bmp",pred_img)
            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            if epoch_idx%1==0:
                save_name = os.path.join(args.result_dir, str(epoch_idx)+'train.bmp')
                pred_clip = torch.clamp(pred_img, 0, 1)
                pred_clip += 0.5 / 255
                pred_img = Fs.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred_img.save(save_name)
            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)

        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
