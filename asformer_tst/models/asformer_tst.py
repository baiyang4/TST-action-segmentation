# Originally written by yabufarha
# https://github.com/yabufarha/ms-tcn/blob/master/model.py

from typing import Any, Optional, Tuple
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numba as nb

import copy
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################
# >>>>>>>>>> pixel decoder - MS-TCN <<<<<<<<<<<<<<<<
#######################################################

class MultiStageTCN(nn.Module):
    """
    Y. Abu Farha and J. Gall.
    MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

    parameters used in originl paper:
        n_features: 64
        n_stages: 4
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.stage1 = SingleStageTCN(in_channel, n_features, n_classes, n_layers)

        stages = [
            SingleStageTCN(n_classes, n_features, n_classes, n_layers)
            for _ in range(n_stages - 1)
        ]
        self.stages = nn.ModuleList(stages)

        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # for training
            outputs = []
            out = self.stage1(x)
            outputs.append(out)
            for stage in self.stages:
                out = stage(self.activation(out))
                outputs.append(out)
            return outputs
        else:
            # for evaluation
            out = self.stage1(x)
            for stage in self.stages:
                out = stage(self.activation(out))
            return out


class SingleStageTCN(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        feature_all = []
        for layer in self.layers:
            out = layer(out)
            feature_all.append(out) # [bs, dim, T]
        out = self.conv_out(out)
        return out, feature_all


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channel: int, out_channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return x + out


class NormalizedReLU(nn.Module):
    """
    Normalized ReLU Activation prposed in the original TCN paper.
    the values are divided by the max computed per frame
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x /= x.max(dim=1, keepdim=True)[0] + self.eps

        return x

#######################################################
# >>>>>>>>>> Asformer <<<<<<<<<<<<<<<<
#######################################################
def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape
        
        assert c1 == c2
        
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6) # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention) 
        attention = attention * padding_mask
        attention = attention.permute(0,2,1)
        out = torch.bmm(proj_val, attention)
        return out, attention

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type): # r1 = r2
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder','decoder']
        
        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()
        
    
    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2)
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2* (self.bl //2)))
        for i in range(self.bl):
            window_mask[:, :, i:i+self.bl] = 1
        return window_mask.to(device)
    
    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder
        
        query = self.query_conv(x1)
        key = self.key_conv(x1)
         
        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)
            
        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    
    def _normal_self_att(self,q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
        
    def _block_wise_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb,1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)
        
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        
        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
    
    def _sliding_window_self_att(self, q,k,v, mask):
        # block operation
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()
        
        
        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl 
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)
        
        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        
        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k, torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v, torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask, torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)
        
        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # special case when self.bl = 1
        v = torch.cat([v[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) 
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat([padding_mask[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask 
        
        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
#         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out
            

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )
        
    def forward(self, x):
        return self.layer(x)
    

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1) # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)
#         self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
#         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in # 2**i
             range(num_layers)])
        
#         self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
#         feature = self.position_en(feature)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
#         out = self.conv_out(feature) * mask[:, 0:1, :]

        return feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()#         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature
    
class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='sliding_att', alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att', alpha=exponential_descrease(s))) for s in range(num_decoders)]) # num_decoders
        self.activation = nn.Softmax(dim=1)
        
    def forward(self, x, mask):
        outputs = []
        out, feature = self.encoder(x, mask)
        outputs.append(self.activation(out))
        
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature* mask[:, 0:1, :], mask)
            outputs.append(self.activation(out))
 
        return outputs


#######################################################
# >>>>>>>>>> segment decoder - local SA <<<<<<<<<<<<<<<<
#######################################################
class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # add
        self.softmax = nn.Softmax(dim=-1)
        self.conv_out = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def generate_attention_mask(self, num_seg, rate):
        
        attention_mask = np.zeros((num_seg, num_seg))
        for idx in range(num_seg):
            start_idx = idx - rate
            end_idx = idx + rate
            if start_idx < 0:
                start_idx = 0
            elif end_idx > num_seg:
                end_idx = num_seg
            attention_mask[idx][start_idx:end_idx+1] = 1

        return attention_mask

    def forward(self, src, src_mask = None, src_key_padding_mask = None, pos = None, rate=None):
        q = k = self.with_pos_embed(src, pos) # q/k/src: [bs, num_seg, dim]
        q = q.permute(0, 2, 1) # [bs, dim, num_seg]
        k = k.permute(0, 2, 1)
        v = src.permute(0, 2, 1)

        # local attention
        _, dim, num_seg = q.shape

        attention = torch.bmm(q.permute(0, 2, 1), k) / np.sqrt(dim) # [bs, num_seg, num_seg]
        attention_mask = self.generate_attention_mask(num_seg, rate)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(src.device) # [num_seg, num_seg]
        attention[~attention_mask.bool()] = float('-inf')

        attention = self.softmax(attention).permute(0, 2, 1) 
        src2 = torch.bmm(v, attention)
        src2 = self.conv_out(F.relu(src2)) # [bs, dim, num_seg]
        src2 = src2.permute(0, 2, 1)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


#######################################################
# >>>>>>>>>> segment decoder - local CA <<<<<<<<<<<<<<<<
#######################################################
class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.tgt2_linear = nn.Linear(d_model, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.softmax = nn.Softmax(dim=-1)

    def generate_attention_mask(self, num_seg, T, action_idx):
        # step1: 生成 seg_t: [num_seg, T]
        action_idx = action_idx.tolist()
        prev_action_class = action_idx[0]
        seg_t = []

        start_id = 0
        for idx, action_class in enumerate(action_idx):
            if action_class != prev_action_class:
                end_id = idx
                tmp = torch.zeros(T)
                tmp[start_id:end_id] = 1
                seg_t.append(tmp)
                prev_action_class = action_class
                start_id = idx
        tmp = torch.zeros(T)
        tmp[start_id:T] = 1
        seg_t.append(tmp)
        seg_t = torch.stack(seg_t, dim = 0)   # [num_seg, T]

        # step2: 利用seg_t造mask
        attention_mask = torch.zeros((num_seg, T)).to(seg_t.device)
        for idx in range(num_seg):
            if idx == 0:
                curr_t = seg_t[idx]
                next_t = seg_t[idx+1]
                attention_mask[idx] = curr_t + next_t
            elif idx == num_seg-1:
                curr_t = seg_t[idx]
                pre_t = seg_t[idx-1]
                attention_mask[idx] = pre_t + curr_t
            else:
                pre_t = seg_t[idx-1]
                curr_t = seg_t[idx]
                next_t = seg_t[idx+1]
                attention_mask[idx] = pre_t + curr_t + next_t
        return attention_mask.unsqueeze(0)

    def forward(self, tgt, memory, pos = None, query_pos = None, action_idx=None):

        q = tgt + query_pos # [num_seg, bs, dim]
        k = memory + pos  # [T, bs, dim]
        v = memory # # [T, bs, dim]
        num_seg, _, dim = q.shape
        T = k.shape[0]
        attention = torch.bmm(q.permute(1, 0, 2), k.permute(1, 2, 0))/np.sqrt(dim) # [bs, num_seg, T]
        attention_mask = self.generate_attention_mask(num_seg, T, action_idx) # [bs, num_seg, T]
        attention[~attention_mask.bool()] = float('-inf')

        attention = self.softmax(attention) # [bs, num_seg, T]
        tgt2 = torch.bmm(v.permute(1, 2, 0), attention.permute(0, 2, 1)) # [bs, dim, T] x [bs, T, num_seg] = [bs, dim, num_seg]
        tgt2 = self.tgt2_linear(F.relu(tgt2.permute(0, 2, 1))).permute(1, 0, 2) # [bs, num_seg, dim]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt # [num_seg, 1, dim*2]

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def get_sinusoid_encoding_table(n_position, d_hid):

    @nb.jit()
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) # T
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table)
            


#######################################################
# >>>>>>>>>> asrf model <<<<<<<<<<<<<<<<
#######################################################
class ActionSegmentRefinementFramework(nn.Module):
    def __init__(self, in_channel, n_features, n_classes, n_stages, n_layers, n_stages_asb, n_stages_brb, **kwargs):

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()

        #######################################################
        # >>>>>>>>>> shared backbone <<<<<<<<<<<<<<<<
        #######################################################
        self.shared_layers = Encoder(n_layers, 2, 2, n_features, 2048, n_classes, 0.3, att_type='sliding_att', alpha=1)
        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)

        #######################################################
        # >>>>>>>>>> action segmentation branch <<<<<<<<<<<<<<<<
        #######################################################
        asb = [
            copy.deepcopy(Decoder(n_layers, 2, 2, n_features, n_classes, n_classes, att_type='sliding_att', alpha=exponential_descrease(s))) for s in range(n_stages_asb - 1)
        ]

        #######################################################
        # >>>>>>>>>> boundary regression branch <<<<<<<<<<<<<<<<
        #######################################################
        brb = [
            SingleStageTCN(1, n_features, 1, n_layers) for _ in range(n_stages_brb - 1)
        ]

        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
 
        #######################################################
        # >>>>>>>>>> shared backbone <<<<<<<<<<<<<<<<
        #######################################################
        mask = torch.full(x.shape, 1, device=device, dtype=torch.float16)
        feature = self.shared_layers(x, mask)
        out_cls = self.conv_cls(feature)     # [bs, num_cls, T]
        out_bound = self.conv_bound(feature) # [bs, 1, T]

        outputs_cls = [out_cls]
        outputs_bound = [out_bound]

        #######################################################
        # >>>>>>>>>> action segmentation branch <<<<<<<<<<<<<<<<
        #######################################################
        for as_stage in self.asb:
            out_cls, as_f = as_stage(self.activation_asb(out_cls)* mask[:, 0:1, :], feature*mask[:,0:1, :], mask)
            outputs_cls.append(out_cls)
        
        #######################################################
        # >>>>>>>>>> boundary regression branch <<<<<<<<<<<<<<<<
        #######################################################
        for br_stage in self.brb:
            out_bound, br_f = br_stage(self.activation_brb(out_bound)) # [bs, 1, T]
            outputs_bound.append(out_bound)

        return outputs_cls, outputs_bound, as_f, br_f



class ourmodel(nn.Module):
    def __init__(self, args, backbone, n_features, n_classes, n_stages, n_layers, n_stages_asb, n_stages_brb, **kwargs):
        super().__init__()
        self.stage2 = args.stage2
        self.stage3 = args.stage3
        print("*************stage2, stage3:", self.stage2, self.stage3)
        self.n_classes = n_classes
        
        #######################################################
        # >>>>>>>>>> shared backbone <<<<<<<<<<<<<<<<
        #######################################################
        self.backbone = backbone
        
        #######################################################
        # >>>>>>>>>> action segmentation branch <<<<<<<<<<<<<<<<
        #######################################################
        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        # >>>>>>> 2.1 action segmentation branch
        # >>>>>>> 2.1.1 pixel decoder
        self.pixel_decoder = SingleStageTCN(n_features, n_features, n_classes, n_layers)
        self.pd_linear1 = MLP(64, 256, 512, 3)

        # >>>>>>> 2.1.2 segment decoder
        self.sd_dim = 256
        self.sd_linear1 = nn.Linear(n_features, self.sd_dim)
        self.sd_linear2 = MLP(19, 128, self.sd_dim, 3)
        self.label_embedding = nn.Embedding(n_classes, self.sd_dim)

        # step3: cross-attention/ self-attention
        
        self.local_ca = CrossAttention(self.sd_dim*2, 1, self.sd_dim*2, 0.1, 'relu')
        self.local_sa = SelfAttention(self.sd_dim*2, nhead=1, dim_feedforward=self.sd_dim*2, dropout = 0.1)
        self.rate = 4
        print("*************head: localsa_rate", 1, self.rate)

        self.class_embed = nn.Linear(self.sd_dim*2, n_classes)
        self.mask_embed = nn.Linear(self.sd_dim*2, n_features)

        #######################################################
        # >>>>>>>>>> boundary regression branch <<<<<<<<<<<<<<<<
        #######################################################
        brb = [SingleStageTCN(1, n_features, 1, n_layers) for _ in range(n_stages_brb - 1)]
        self.brb = nn.ModuleList(brb)
        self.activation_brb = nn.Sigmoid()


    def forward(self, x, batch_target=None, batch_targets_segment=None, location_segment=None):
        #######################################################
        # >>>>>>>>>> shared backbone <<<<<<<<<<<<<<<<
        #######################################################
        # out_cls: [bs, num_cls, T]. out_bound: [bs, 1, T]
        out_cls, out_bound, as_f_all, _ = self.backbone(x) 
        bk_f = as_f_all # [bs, bk_dim, T]

        #######################################################
        # >>>>>>>>>> our model<<<<<<<<<<<<<<<<
        #######################################################
        # >>>>>>> 2.1.1 pixel decoder
        _, pd_f_all = self.pixel_decoder(bk_f) # [bs, bk_dim, T]
        pd_f_9 = pd_f_all[-2]
        pd_f_10 = pd_f_all[-1]
        pd_f_memory = self.pd_linear1(pd_f_9.permute(0, 2, 1)).permute(0, 2, 1) # [bs, sd_dim*2, T]

        if self.stage2:
            if self.training:
                action_idx = batch_target[0]
            else:
                action_idx = torch.max(out_cls[-1][0], 0)[1]
        elif self.stage3:
            action_idx = torch.max(out_cls[-1][0], 0)[1]

        bk_f = self.sd_linear1(bk_f.permute(0, 2, 1)).permute(0, 2, 1) # [bs, sd_dim, T]
        # segment_feat: [bs, num_seg, sd_dim]. segment_onehot:[bs, num_seg, num_class]. PREDlabel_list:[bs, num_seg]
        segment_idx, segment_feat, segment_onehot, PREDlabel_list, GTlabel_list, GTmask_list = self.get_segment_info(action_idx, bk_f, batch_target, batch_targets_segment, location_segment, self.training)
        segment_onehot = self.sd_linear2(segment_onehot) # [bs, num_seg, sd_dim]
        segment_feat = segment_feat + segment_onehot

        num_seg = segment_feat.shape[1]
        label_embed = self.label_embedding(PREDlabel_list) # [bs, num_seg, dim]
        segment_feat = torch.cat([segment_feat, label_embed], dim=2).permute(1, 0, 2) # [bs, num_seg, seg_dim*2] => [num_seg, bs, seg_dim*2]
       
        # >>>>>>>>>>>>>>> Cross-attention: query(refine_input). key/value(refine_input)
        # >>> key/value(pixel decoder)
        pd_f_memory = pd_f_memory.permute(2, 0, 1) # [T, bs, sd_dim*2]
        T, _, input_dim = pd_f_memory.shape
        pos = get_sinusoid_encoding_table(T, input_dim).to(x.device).unsqueeze(1) # [T, 1, sd_dim*2]

        # >>>> query(segment decoder)
        tgt = torch.zeros_like(segment_feat).to(x.device) # [num_seg, bs, sd_dim*2]
        tgt = self.local_ca(tgt=tgt, query_pos=segment_feat, memory=pd_f_memory, pos=pos, action_idx=action_idx).permute(1, 0, 2)  # [num_seg, bs, sd_dim*2] => [bs, num_seg, sd_dim*2]

        # >>>>>>>>>>>>>>>> local-self attention
        _, num_seg, hidden_dim = tgt.shape
        pos = get_sinusoid_encoding_table(num_seg, hidden_dim).to(x.device) # [num_seg, sd_dim*2]
        refine_pred = self.local_sa(src=tgt, src_mask=None, src_key_padding_mask=None, pos=pos, rate=self.rate) # [bs, num_seg, sd_dim*2]
       
        segment_cls = self.class_embed(refine_pred) # [bs, num_seg, num_class]
        mask_embed = self.mask_embed(refine_pred) # [bs, num_seg, bk_dim]
        segment_mask = torch.bmm(mask_embed, pd_f_10) # [bs, num_seg, T]
        return out_cls, out_bound, segment_cls, segment_mask, GTlabel_list, GTmask_list, self.stage3

    def get_segment_info(self, action_idx, batch_input, batch_target=None, batch_targets_segment=None, location_segment=None, flag=None): 
        # action_idx: [T]
        # batch_input: [bs, sd_dim, T]
        # batch_target: [1, T]
        # batch_targets_segment：[num_seg, T]

        segment_idx = [0]
        prev_seg = action_idx[0]
        for ii, idx in enumerate(action_idx):
            if idx != prev_seg:
                segment_idx.append(ii)
            prev_seg = idx
        segment_idx.append(len(action_idx))
        
        GTlabel_list = list()
        PREDlabel_list = list()
        segment_feat = list()
        GTmask_list = list()
        segment_onehot = list()

        for s_i in range(len(segment_idx)-1):
            prev_idx = segment_idx[s_i]
            curr_idx = segment_idx[s_i+1]
            curr_seg = batch_input[:, :, prev_idx:curr_idx]

            if flag:
                T = batch_target.shape[-1]
                # 构造seg的cls
                GTseg = batch_target[:, prev_idx:curr_idx]
                GTseg_label = torch.argmax(torch.bincount(GTseg[0]))
                GTlabel_list.append(GTseg_label)
                # 构造seg的mask
                firset_offset = torch.where(GTseg == GTseg_label)[0][0]
                local = prev_idx + firset_offset
                for idx, sub_interval in enumerate(location_segment):
                    start_idx = torch.tensor(sub_interval[0]).to(batch_input.device)
                    end_idx = torch.tensor(sub_interval[1]).to(batch_input.device)
                    if start_idx <= local < end_idx:
                        GTmask_list.append(batch_targets_segment[idx])
                        break
                    elif local == T-1:
                        GTmask_list.append(batch_targets_segment[-1])
                        break

            PREDseg_label = torch.mean(action_idx[prev_idx:curr_idx].float()).long()
            PREDlabel_list.append(PREDseg_label)
            
            curr_onehot = torch.zeros(self.n_classes)
            curr_onehot[PREDseg_label] = 1

            curr_seg = curr_seg.mean(-1, True).permute(0, 2, 1) # [1, 1, sd_dim]
            segment_feat.append(curr_seg) 
            segment_onehot.append(curr_onehot)
            
        GTlabel_list = torch.LongTensor(GTlabel_list).view(1, -1).to(batch_input.device) # [bs, num_seg]
        PREDlabel_list = torch.LongTensor(PREDlabel_list).view(1, -1).to(batch_input.device) # [bs, num_seg]
        segment_feat = torch.cat(segment_feat, dim=1) # [bs, num_seg, dim]
        if flag:
            GTmask_list = torch.stack(GTmask_list) # [num_seg, T]
        else:
            GTmask_list = None
        segment_onehot = torch.stack(segment_onehot).unsqueeze(0).to(batch_input.device) # [bs, num_seg, num_class]
        return segment_idx, segment_feat, segment_onehot, PREDlabel_list, GTlabel_list, GTmask_list
