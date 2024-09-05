from dataset.dataset import TestSet
from model import Vgg19
from skimage import metrics
import os
import numpy as np
from imageio import imread, imsave
from PIL import Image

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd # for saving losses
from timeit import default_timer as timer  # for recording
from pathlib import Path
from utils import StitchAndCalJson, randomSize
from torch.utils.data import DataLoader

def process(img):
    img = (img+1.) * 127.5
    img = img.squeeze().round().cpu().numpy().astype(np.uint8)
    return img
class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        # print info
        self.logger.info("Current path: "+os.getcwd())
        if not (self.args.eval or self.args.test):
            args_dict = vars(self.args)
            for key in args_dict.keys():
                self.logger.info("{}: {}".format(key, args_dict[key]))
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(self.vgg19, list(range(self.args.num_gpu)))

        self.params1 = [
            {"params": filter(lambda p: p.requires_grad, self.model.Decoder.parameters() if 
             args.num_gpu==1 else self.model.module.Decoder.parameters()),
             "lr": args.lr_rate
            },
            {"params": filter(lambda p: p.requires_grad, self.model.Encoder.parameters() if 
             args.num_gpu==1 else self.model.module.Encoder.parameters()), 
             "lr": args.lr_rate_lte
            }
        ]
        self.params2 = [
            {"params": filter(lambda p: p.requires_grad, self.model.LTG.parameters() if
                              args.num_gpu == 1 else self.model.module.LTG.parameters()),
             "lr": args.ref_rate
             }
        ]
        self.optimizer = []
        self.scheduler = []
        self.optimizer.append(optim.Adam(self.params1, betas=(args.beta1, args.beta2), eps=args.eps))
        self.scheduler.append(optim.lr_scheduler.StepLR(
            self.optimizer[0], step_size=self.args.decay, gamma=self.args.gamma))
        
        self.optimizer.append(optim.Adam(self.params2, betas=(
            args.beta1, args.beta2), eps=args.eps))
        self.scheduler.append(optim.lr_scheduler.StepLR(
            self.optimizer[1], step_size=self.args.decay, gamma=self.args.gamma))
        
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

        # ---- load pre-trained model
        try:
            self.load(args.model_path)
        except:
            self.logger.info("Load model fail")


        self.logger.info('# para: %d' % sum(param.numel()
                                            for param in self.model.parameters() if param.requires_grad))
        if not (self.args.test or self.args.eval):
            self.logger.info("batch size: %d, # batches: %d"%(self.args.batch_size, len(self.dataloader['train'])))
    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            #model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict, strict = True)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, is_init=False):
        self.model.train()
        if (not is_init):
            self.scheduler[0].step()
            self.scheduler[1].step()
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer[0].param_groups[0]['lr']))
        self.logger.info('Current epoch learning rate -2: %e' %(self.optimizer[1].param_groups[0]['lr']))
        losses = {}
        losses['loss'] = 0
        losses['rec_loss'] = 0
        losses['per_loss'] = 0
        losses['ref_loss'] = 0
        losses['tpl_loss'] = 0
        losses['adv_loss'] = 0
        losses['time'] = timer()
        cnt = 0
        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            # if(len(self.losses['loss']) > 10):
            #     pd.DataFrame.from_dict(self.losses).to_csv(self.args.save_dir+'/losses.csv', mode='a', header=False)
            #     for key in self.losses.keys():
            #         self.losses[key] = []
            self.optimizer[0].zero_grad()
            self.optimizer[1].zero_grad()
            cnt += 1
            sample_batched = self.prepare(sample_batched)
            lr = sample_batched['LR']
            lr_sr = sample_batched['LR_sr']
            hr = sample_batched['HR']
            ref = sample_batched['Ref']
            ref_sr = sample_batched['Ref_sr']
            sr, S, T_lv3, T_lv2, T_lv1, rec_ref_lv1, rec_ref_lv2, rec_ref_lv3 = self.model(
                lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)  # T_lv3,

            ### calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0) ### flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            loss = rec_loss
            losses['rec_loss'] += loss.item()
            if (is_print):
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.10f' %(rec_loss.item()) )

            if('ref_loss') in self.loss_all:
                ref_loss = self.args.ref_w * self.loss_all['ref_loss'](rec_ref_lv3, rec_ref_lv2, rec_ref_lv1,
                                                                        S, T_lv3, T_lv2, T_lv1)
                loss += ref_loss
                losses['ref_loss'] += ref_loss.item()
                if (is_print):
                    self.logger.info( 'ref_loss: %.10f' %(ref_loss.item()) )  
            if ('per_loss' in self.loss_all):
                sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                with torch.no_grad():
                    hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                per_loss = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)
                loss += per_loss
                # self.losses['per_loss'].append(per_loss.item())
                losses['per_loss'] += per_loss.item()
                if (is_print):
                    self.logger.info( 'per_loss: %.10f' %(per_loss.item()) )
            if ('tpl_loss' in self.loss_all):
                sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr)
                tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1,
                                                                        S, T_lv3, T_lv2, T_lv1)
                loss += tpl_loss
                # self.losses['tpl_loss'].append(tpl_loss.item())
                losses['tpl_loss'] += tpl_loss.item()
                if (is_print):
                    self.logger.info( 'tpl_loss: %.10f' %(tpl_loss.item()) )    
            if ('adv_loss' in self.loss_all):
                adv_loss = self.args.adv_w * self.loss_all['adv_loss'](sr, hr)
                loss += adv_loss
                # self.losses['adv_loss'].append(adv_loss.item())
                losses['adv_loss'] += adv_loss.item()
                if (is_print):
                    self.logger.info( 'adv_loss: %.10f' %(adv_loss.item()) )
            losses['loss'] += loss.item()
            loss.backward()
            self.optimizer[0].step()
            self.optimizer[1].step()
        
        losses['loss'] /= cnt
        losses['rec_loss'] /= cnt
        losses['per_loss'] /= cnt
        losses['adv_loss'] /= cnt
        losses['ref_loss'] /= cnt
        losses['time'] = timer() - losses['time']
        save_dir = Path(self.args.save_dir)
        if current_epoch == 1:
            pd.DataFrame(losses, index=[current_epoch]).to_csv(save_dir / 'loss.csv')
        else:
            pd.DataFrame(losses, index=[current_epoch]).to_csv(save_dir / 'loss.csv', mode='a', header=False)

        self.logger.info(('init ' if is_init else '') + '%d total loss: %.10f' %
                         (current_epoch, losses['loss']))
        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    def test(self):
        gt = self.args.test_gt
        data_root = self.args.dataset_dir
        meta = randomSize().createJsonFile(data_root)
        test_dataloader = DataLoader(TestSet(data_root, meta=meta, has_gt=self.args.test_gt),batch_size=1, shuffle=False)
        save_path = Path(self.args.save_dir) / 'save_results'
        self.logger.info("Start testing. ")
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(test_dataloader):
                sample_batched = self.prepare(sample_batched)
                lr = sample_batched['LR']
                lr_sr = sample_batched['LR_sr']
                sr = self.model(lr=lr, lrsr=lr_sr)             
                lr = process(lr); sr = process(sr)
            
                imsave(save_path / (str(i_batch).zfill(5)+'_sr.png'), sr)
                imsave(save_path / (str(i_batch).zfill(5)+'_lr.png'), lr)
                if gt:
                    hr = sample_batched['HR']
                    hr = process(hr); 
                    imsave(save_path / (str(i_batch).zfill(5)+'_hr.png'), hr)

            if gt:
                res = StitchAndCalJson().stitchAndCal(path=save_path, meta=meta, cats=['lr','hr','sr'])
                self.logger.info("PSNR: %.4f, SSIM: %.4f" % (res['psnr'], res['ssim']))
            else:
                StitchAndCalJson().stitch(path=save_path, meta=meta, cats=['lr','sr'])
        self.logger.info("Testing over. ")
            


    def evaluate(self, current_epoch=0):
        
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')

        self.model.eval()
        lst = []
        with torch.no_grad():
            psnr, ssim, cnt = 0., 0., 0
            for i_batch, sample_batched in enumerate(self.dataloader['val']):
                cnt += 1
                sample_batched = self.prepare(sample_batched)
                lr = sample_batched['LR']
                lr_sr = sample_batched['LR_sr']
                hr = sample_batched['HR']
                sr = self.model(lr=lr, lrsr=lr_sr)
                
                
                hr = process(hr); lr = process(lr); sr = process(sr)
                if (self.args.eval_save_results):
                    imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'_sr.png'), sr)
                    imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'_hr.png'), hr)
                    imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'_lr.png'), lr)
                
                _psnr = metrics.peak_signal_noise_ratio(hr, sr)
                _ssim = metrics.structural_similarity(hr, sr, multichannel=False)
                psnr += _psnr
                ssim += _ssim

            psnr_ave = psnr / cnt
            ssim_ave = ssim / cnt
            # self.logger.info('PSNR (now): %.3f \t SSIM (now): %.4f' %(psnr_ave, ssim_ave))
            lst += [psnr_ave, ssim_ave]
            if (psnr_ave > self.max_psnr):
                self.max_psnr = psnr_ave
                self.max_psnr_epoch = current_epoch
                if current_epoch > 0:
                    self.logger.info('saving the best psnr model...')
                    tmp = self.model.state_dict()
                    model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                        (('SearchNet' not in key) and ('_copy' not in key))}
                    model_name = self.args.save_dir+'/model/model_bst_psnr'+'.pth'
                    torch.save(model_state_dict, model_name)
            if (ssim_ave > self.max_ssim):
                self.max_ssim = ssim_ave
                self.max_ssim_epoch = current_epoch
                if current_epoch > 0:
                    self.logger.info('saving the best ssim model...')
                    tmp = self.model.state_dict()
                    model_state_dict = {key.replace('module.', ''): tmp[key] for key in tmp if
                                        (('SearchNet' not in key) and ('_copy' not in key))}
                    model_name = self.args.save_dir+'/model/model_best_ssim'+'.pth'
                    torch.save(model_state_dict, model_name)
        self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)'
                         % (self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))
        save_dir = Path(self.args.save_dir)
        if current_epoch == 1:
            pd.DataFrame({'PSNR': psnr, "SSIM": ssim}, index=[current_epoch]).to_csv(save_dir / 'eval.csv', header=True)
        else:
            pd.DataFrame({'PSNR': psnr, "SSIM": ssim}, index=[current_epoch]).to_csv(save_dir / 'eval.csv', mode='a', header=False)
        if self.args.eval_save_results:
            from ImageStitcher import StitchAndCalFlex
            res = StitchAndCalFlex().stitchAndCal(save_dir / 'save_results',3,160,290,['hr','lr','sr'])
        # self.logger.info("PSNR: {}, SSIM: {}".format(res['psnr'], res['ssim']))
        self.logger.info("PSNR: {}, SSIM: {}".format(psnr_ave, ssim_ave))
        self.logger.info('Evaluation over.')
