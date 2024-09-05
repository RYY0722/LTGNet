from model import Encoder, Decoder, SearchTransfer, LTG

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTGNet(nn.Module):
    def __init__(self, args):
        super(LTGNet, self).__init__()
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )
        self.Decoder = Decoder.Decoder(num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, 
            res_scale=args.res_scale)
        self.Encoder      = Encoder.Encoder(requires_grad=True)
        self.Encoder_copy = Encoder.Encoder(requires_grad=False)
        self.SearchTransfer = SearchTransfer.SearchTransfer()
        self.LTG = LTG.LTG(64, num_res_blocks=self.args.num_MSFP)

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None): 
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.Encoder_copy.load_state_dict(self.Encoder.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.Encoder_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3
        lrsr_lv1, lrsr_lv2, _  = self.Encoder((lrsr.detach() + 1.) / 2.)
        lr_lv1, lr_lv2, lr_lv3 = self.Encoder((lr.detach() + 1.) / 2.)
        generated_tex_lv3, generated_tex_lv2, generated_tex_lv1 = self.LTG(lr_lv1=lr_lv1, lr_lv2=lr_lv2, lr_lv3=lr_lv3)
        training = ref is not None
        if training: 
            _, refsr_lv2, _ = self.Encoder((refsr.detach() + 1.) / 2.)

            ref_lv1, ref_lv2, ref_lv3 = self.Encoder((ref.detach() + 1.) / 2.)
            
            S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv2, refsr_lv2, ref_lv1, ref_lv2, ref_lv3)
            T_lv1 = T_lv1 * F.interpolate(S, scale_factor=4, mode='bicubic')
            T_lv2 = T_lv2 * F.interpolate(S, scale_factor=2, mode='bicubic')
            T_lv3 = T_lv3 * S
        else:
            T_lv1 = generated_tex_lv1
            T_lv2 = generated_tex_lv2
            T_lv3 = generated_tex_lv3
        sr = self.Decoder(lr, lrsr_lv1, lrsr_lv2, T_lv3, T_lv2, T_lv1)
        sr = torch.clamp(sr, -1, 1)
        if training:
            return sr, S, T_lv3, T_lv2, T_lv1, generated_tex_lv1, generated_tex_lv2, generated_tex_lv3
        else:
            return sr
