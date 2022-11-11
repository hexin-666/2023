import os
import torch
from cv2 import cv2
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time


def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir,args.data_dir1, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()

        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, input_img1, label_img1,name = data
            input_img = input_img.to(device)
            input_img1 = input_img1.to(device)
            tm = time.time()
            _ = model(input_img,input_img1)
            _ = time.time() - tm

            if iter_idx == 20:
                break

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, input_img, label_img,name = data

            input_img = input_img.to(device)
            input_img1 = input_img1.to(device)

            tm = time.time()

            pred = model(input_img,input_img1)

            elapsed = time.time() - tm
            adder(elapsed)


            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()
            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)



            if args.save_image:
                save_name = os.path.join(args.result_dir,name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred = pred.resize((141,329))
                pred.save(save_name)


            psnr_adder(psnr)
            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print("Average time: %f" % adder.average())
