# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 12:42:24 2021

@author: pelb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7,act=nn.ReLU(),bias=True):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation,bias=bias),
            #nn.BatchNorm1d(channel_out),
            act
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2,act=nn.ReLU(),bias=True):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding,bias=bias),
            #nn.BatchNorm1d(channel_out),
            act
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer_T(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2,act=nn.ReLU(),bias=True):
        super(UpSamplingLayer_T, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding,bias=bias),
            #nn.BatchNorm1d(channel_out),
            act
        )

    def forward(self, ipt):
        return self.main(ipt)




class waveEar(nn.Module):
    def __init__(self, n_layers=4, num_filters=128,kernel_size=65,out_chans=201,act=nn.ReLU(),act_decoder=None,bias=True,bias_out=False,PReLU_chan=False,channels_in=1):
        super(waveEar, self).__init__()
        self.act = act
        self.act_decoder = act_decoder
        self.n_layers = n_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self
        #self.padding = int(np.floor((kernel_size-1)+1-2)/2)
        self.padding = int(np.floor(kernel_size/2))
        self.out_chans = out_chans
        encoder_in_channels_list = [channels_in] + [self.num_filters for i in range(1,self.n_layers)]
        encoder_out_channels_list = [self.num_filters for i in range(1, self.n_layers + 1)]
        self.bias = bias
        self.bias_out = bias_out
        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    act = self.act,
                    bias = self.bias
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.num_filters, self.num_filters, self.kernel_size, stride=1,
                      padding=self.padding,bias=self.bias),
            #nn.BatchNorm1d(self.n_layers * self.channels_interval),
            self.act
        )

        decoder_in_channels_list = [2*self.num_filters for i in range(0, self.n_layers)]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        decoder_out_channels_list[-1] = 2*self.num_filters
        #decoder_out_channels_list[-1] = self.out_chans
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            if str(act_decoder) == str(nn.PReLU()):
                act_p = torch.nn.PReLU(decoder_out_channels_list[i])
            else:
                act_p = act_decoder
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    #if PReLU_chan == True:
                    #act = torch.nn.PReLU(decoder_out_channels_list[i]),
                   # else:
                    act = act_p,
                    bias = self.bias
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(channels_in + 2* self.num_filters, self.out_chans, kernel_size=1, stride=1,bias=self.bias_out)
        )

    def forward(self, input):
        tmp = []
        o = input

        # Up Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]
        o = self.middle(o)
        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o



class CoNNEar(nn.Module):
    def __init__(self, n_layers=4, num_filters=128,kernel_size=64,out_chans=201,act=nn.ReLU(),act_decoder=None,bias=True,bias_out=False,PReLU_chan=False,channels_in=1):
        super(CoNNEar, self).__init__()
        self.act = act
        self.act_decoder = act_decoder
        self.n_layers = n_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = 2
        self.padding = int(np.floor((kernel_size-1)+1-2)/2)
        self.out_chans = out_chans
        encoder_in_channels_list = [channels_in] + [self.num_filters for i in range(1,self.n_layers)]
        encoder_out_channels_list = [self.num_filters for i in range(1, self.n_layers + 1)]
        self.bias = bias
        self.bias_out = bias_out
        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    act = self.act,
                    bias = self.bias,
                    stride = self.stride
                )
            )

        decoder_in_channels_list = [self.num_filters] + [2*self.num_filters for i in range(1, self.n_layers)]
      #  decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        decoder_out_channels_list[-1] = self.out_chans
        #decoder_out_channels_list[-1] = self.out_chans
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            if str(act_decoder) == str(nn.PReLU()):
                act_p = torch.nn.PReLU(decoder_out_channels_list[i])
            else:
                act_p = act_decoder
            self.decoder.append(
                UpSamplingLayer_T(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                    kernel_size = self.kernel_size,
                    padding = self.padding,
                    #if PReLU_chan == True:
                    #act = torch.nn.PReLU(decoder_out_channels_list[i]),
                   # else:
                    act = act_p,
                    bias = self.bias,
                    stride = self.stride
                )
            )


    def forward(self, input):
        tmp = []
        o = input
        # Up Sampling
        for i in range(self.n_layers-1):
            o = self.encoder[i](o)
            tmp.append(o)
        o = self.encoder[i+1](o)
        o = self.decoder[0](o)
        for i in range(1,self.n_layers):
            o = torch.cat([o, tmp[self.n_layers - i-1]], dim=1)
            o = self.decoder[i](o)
        return o