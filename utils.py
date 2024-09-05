import logging
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import json
import numpy as np
from PIL import Image
from imageio import imread
import glob, os
from pathlib import Path
import shutil
import pandas as pd

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import shutil
import json
import matplotlib.pyplot as plt
# preprocessing
class randomSize():
    def __init__(self):
        super(randomSize, self).__init__()
    def myglob(self, path, pat):
        return sorted([file for file in path.glob(pat)])
    def createJsonFile(self, path=None, patch_size=160, stride=65):
        path = Path(path)
        dic = dict()


        hrs = self.myglob(path,'*_lr.png')

        meta_name = 'meta{}s{}.txt'.format(patch_size,stride)
        textfile = open(path / meta_name, "w")
        textfile.write("hr,x1,x2,y1,y2")
        textfile.close()
        
        for hr_name in hrs:
            ID = hr_name.name.split("_lr.png")[0]
            hr = cv2.imread(hr_name.__str__())
            num_x = (hr.shape[0] - patch_size) // stride + 1
            num_y = (hr.shape[1] - patch_size) // stride + 1
            edges = []
            x = -stride
            y = -stride
            for i in range(num_x + 1):
                x = i * stride if i != num_x else hr.shape[0] - patch_size
                for j in range(num_y + 1):        
                    y = j * stride if j != num_y else hr.shape[1] - patch_size
                    edges.append([x, x + patch_size, y, y + patch_size])
                   
            dic[ID] = edges
    
        with open(path / meta_name, 'w') as outfile:
            json.dump(dic, outfile, indent=4)
        return (path / meta_name)


class StitchAndCalJson():
    def __init__(self):
        super(StitchAndCalJson, self).__init__()
    def stitch(self, path, meta, cats):
        # print('path: ',path)
        # print('meta: ',meta)
        with open(meta) as json_file:
            data = json.load(json_file)
        # path = self.cp.run(path)
        try:
            data = data['test']
        except:
            pass
        path = Path(path)
        # print('path: ',path)
        if (path / 'comb').exists():
            shutil.rmtree(path / 'comb')
        (path / 'comb').mkdir(exist_ok=False)
        for cat in cats:
            files = sorted([file for file in path.glob('*_{}.png'.format(cat))])
            i=0
            for ID, lsts in data.items():
                totalX, totalY = lsts[-1][1], lsts[-1][3]
                res = np.zeros((totalX, totalY))
                cnt_map = np.zeros((totalX, totalY))
                # hr = cv2.imread(r'D:\Exercise\Python\datasets\STDR\eehpc\harnet\imgs\{}_hr.png'.format(ID))
                for lst in lsts:
                    x1, x2, y1, y2 = lst
                    # name = "{}_{}_{}_sr.png".format(ID, x1, y1)
                    img = imread(str(files[i]))
                    i+=1
                    try:    
                        res[x1:x2, y1:y2] += img
                        # hr_patch = hr[x1:x2, y1:y2]
                        cnt_map[x1:x2, y1:y2] += 1
                        # fig = plt.figure(figsize=(1,2))
                        # fig.add_subplot(1,2,1)
                        # plt.imshow(img)
                        # fig.add_subplot(1,2,2)
                        # plt.imshow(hr_patch)   
                        # print(lst, 'ok')
                    except:
                        print(img.shape)
                        print(lst, 'not ok')  
                        return
                img = Image.fromarray((res/cnt_map).astype(np.uint8))
                img.save(path / 'comb' / (ID+"_{}.png".format(cat)))
                # break
        return
    def cal(self, path):
        psnr_sum = 0
        ssim_sum = 0
        cnt = 0
        model_psnr = []
        model_ssim = []
        hrs = sorted(glob.glob(os.path.join(path, '*_hr.png')))
        srs = sorted(glob.glob(os.path.join(path, '*_sr.png')))
        assert len(hrs) == len(srs), 'hr != sr'
        for i in range(len(hrs)):
            hr = imread(hrs[i])
            # lr = cv2.imread(lrs[i], cv2.IMREAD_GRAYSCALE)
            sr = imread(srs[i])
            _psnr = psnr(hr, sr)
            _ssim = ssim(hr, sr, multichannel=False)
            psnr_sum += _psnr
            ssim_sum += _ssim
            model_psnr.append(_psnr)
            model_ssim.append(_ssim)
            cnt += 1
            pd.DataFrame(data={'ssim':model_ssim, 'psnr':model_psnr}).to_csv(Path(path).parent / 'res.csv')
        return {'psnr':psnr_sum/cnt,'ssim':ssim_sum/cnt}
    def stitchAndCal(self, path=None, meta=None, cats=None):
        self.stitch(path, meta, cats)
        return self.cal(path / 'comb')

        
        

class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        ### create a logger
        self.__logger = logging.getLogger(logger_name)

        ### set the log level
        self.__logger.setLevel(log_level)

        ### create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        ### create a handler to print on console
        console_handler = logging.StreamHandler()

        ### define the output format of handlers
        formatter = logging.Formatter('[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        ### add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def mkExpDir(args):
    if (os.path.exists(args.save_dir)):
        if (not args.reset):
            raise SystemExit('Error: save_dir "' + args.save_dir + '" already exists! Please set --reset True to delete the folder.')
        else:
            shutil.rmtree(args.save_dir)

    os.makedirs(args.save_dir)
    # os.makedirs(os.path.join(args.save_dir, 'img'))

    if ((not args.eval) and (not args.test)):
        os.makedirs(os.path.join(args.save_dir, 'model'))
    
    if ((args.eval and args.eval_save_results) or args.test):
        os.makedirs(os.path.join(args.save_dir, 'save_results'))

    args_file = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')

    _logger = Logger(log_file_name=os.path.join(args.save_dir, args.log_file_name), 
        logger_name=args.logger_name).get_log()

    return _logger


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False

