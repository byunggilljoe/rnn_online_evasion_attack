import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import split, merge, injective_pad, psi
import torch.backends.cudnn as cudnn
import functools
from torch.nn import init
import random
from itertools import accumulate




class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=4):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.stride = stride
        self.psi = psi(stride)
        layers = []
        if not first:
            layers.append(nn.GroupNorm(1,in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        if int(out_ch//mult)==0:
            ch = 1
        else:
            ch =int(out_ch//mult)
        layers.append(nn.Conv2d(in_ch//2, ch, kernel_size=3,
                      stride=self.stride, padding=1, bias=False))
        layers.append(nn.GroupNorm(1,ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(ch, ch,
                      kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.GroupNorm(1,ch, affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(ch, out_ch, kernel_size=3,
                      padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        x = (x1, x2)
        return x


class STConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, memo_size):
        super(STConvLSTMCell,self).__init__()
        self.KERNEL_SIZE = 3
        self.PADDING = self.KERNEL_SIZE // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memo_size = memo_size
        self.in_gate = nn.Conv2d(input_size + hidden_size + hidden_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.remember_gate = nn.Conv2d(input_size + hidden_size+ hidden_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.cell_gate = nn.Conv2d(input_size + hidden_size+ hidden_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING)

        self.in_gate1 = nn.Conv2d(input_size + memo_size+ hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.remember_gate1 = nn.Conv2d(input_size + memo_size+ hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)
        self.cell_gate1 = nn.Conv2d(input_size + memo_size+ hidden_size, memo_size, self.KERNEL_SIZE, padding=self.PADDING)

        self.w1 = nn.Conv2d(hidden_size + memo_size, hidden_size, 1)
        self.out_gate = nn.Conv2d(input_size + hidden_size +hidden_size+memo_size, hidden_size, self.KERNEL_SIZE, padding=self.PADDING)


    def forward(self, input, prev_state):
        input_,prev_memo = input
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden,prev_cell), 1)

        in_gate = F.sigmoid(self.in_gate(stacked_inputs))
        remember_gate =F.sigmoid(self.remember_gate(stacked_inputs))
        cell_gate = F.tanh(self.cell_gate(stacked_inputs))

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)

        stacked_inputs1 = torch.cat((input_, prev_memo,cell), 1)

        in_gate1 = F.sigmoid(self.in_gate1(stacked_inputs1))
        remember_gate1 = F.sigmoid(self.remember_gate1(stacked_inputs1))
        cell_gate1 = F.tanh(self.cell_gate1(stacked_inputs1))



        memo = (remember_gate1 * prev_memo) + (in_gate1 * cell_gate1)

        out_gate = F.sigmoid(self.out_gate(torch.cat((input_, prev_hidden,cell,memo), 1)))
        hidden = out_gate * F.tanh(self.w1(torch.cat((cell,memo),1)))
        #print(hidden.size())
        return (hidden, cell),memo




class zig_rev_predictor(nn.Module):
    def __init__(self, input_size, hidden_size,output_size,n_layers,batch_size, type = 'lstm', w = 8, h = 8):
        super(zig_rev_predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.type = type
        self.w = w
        self.h = h

        self.convlstm = nn.ModuleList(
                [STConvLSTMCell(input_size, hidden_size,hidden_size) if i == 0 else STConvLSTMCell(hidden_size,hidden_size, hidden_size) for i in
                 range(self.n_layers)])

        self.att = nn.ModuleList([nn.Sequential(nn.Conv2d(self.hidden_size, self.hidden_size, 1, 1, 0),
                                                nn.GroupNorm(1,self.hidden_size,affine=True),
                                                nn.ReLU(),
                                                nn.Conv2d(self.hidden_size, self.hidden_size, 3, 1, 1),
                                                nn.GroupNorm(1, self.hidden_size, affine=True),
                                                nn.ReLU(),
                                                nn.Conv2d(self.hidden_size, self.hidden_size, 1, 1, 0),
                                                nn.Sigmoid()
                                                ) for i in range(self.n_layers)])

        self.hidden = self.init_hidden()
        self.prev_hidden = self.hidden

    def init_hidden(self,cc = 0):
        hidden = []

        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size,self.w,self.h).cuda(cc)),
                           Variable(torch.zeros(self.batch_size, self.hidden_size,self.w,self.h).cuda(cc))))
        return hidden

    # def copy(self,hidden):
    #     self.prev_hidden=[]
    #     if self.type == 'residual':
    #         for i in range(self.n_layers):
    #             self.prev_hidden.append((0,(hidden[i][1][0].clone(),
    #                               hidden[i][1][1].clone())))

    def forward(self, input):
        input_, memo = input
        x1, x2 = input_
        mask = []
        for i in range(self.n_layers):
            out = self.convlstm[i]((x1,memo), self.hidden[i])
            self.hidden[i] = out[0]
            memo = out[1]
            g = self.att[i](self.hidden[i][0])
            mask.append(g)
            x2 = (1 - g) * x2 + g * self.hidden[i][0]
            x1, x2 = x2, x1

        return (x1,x2),memo,mask




class autoencoder(nn.Module):
    def __init__(self, nBlocks, nStrides, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4,it=False):
        super(autoencoder, self).__init__()
        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)
        self.init_ds = init_ds
        if init_ds == 1:
            self.in_ch = in_shape[0]
        else:
            self.in_ch = in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True
        self.it = it
        # print('')
        # print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3))
        if not nChannels:
            nChannels = [self.in_ch//2, self.in_ch//2 * 4,
                         self.in_ch//2 * 4**2, self.in_ch//2 * 4**3]

        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, input, is_predict = True):

        if is_predict:
            n = self.in_ch // 2
            if self.it:
                input = merge(input[0], input[1])
            if self.init_ds != 0:
                x = self.init_psi.forward(input)
            else:
                x = input
            out = (x[:, :n, :, :], x[:, n:, :, :])
            for block in self.stack:
                out = block.forward(out)
            x = out
        else:
            out = input
            for i in range(len(self.stack)):
                out = self.stack[-1 - i].inverse(out)
            out = merge(out[0], out[1])
            if self.init_ds != 0:
                x = self.init_psi.inverse(out)
            else:
                x = out
            if self.it:
                n = self.in_ch // 2
                x = (x[:, :n, :, :], x[:, n:, :, :])
        return x
