# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor



from ltr.models.row_column_decoupled_attention import MultiheadRCDA

class FeatureFusionNetwork(nn.Module):
    def __init__(self, d_model=256, nhead=1,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.,
                 activation="relu", num_feature_levels=1,num_query_position = 1024,num_query_pattern=1,
                 spatial_prior="grid",attention_type="nn.MultiheadAttention"):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.attention_type = attention_type
        encoder_layer = TransformerEncoderLayerSpatial(d_model, dim_feedforward,
                                                          dropout, activation, nhead , attention_type)
        #encoder_layer_level = TransformerEncoderLayerLevel(d_model, dim_feedforward,
            #                                              dropout, activation, nhead)

        decoder_layer = TransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation, nhead,
                                                          num_feature_levels, attention_type)

        if num_feature_levels == 1:
            self.num_encoder_layers_level = 0
        else:
            self.num_encoder_layers_level = num_encoder_layers // 2
        self.num_encoder_layers_spatial = num_encoder_layers - self.num_encoder_layers_level

        self.encoder_layers = _get_clones(encoder_layer, self.num_encoder_layers_spatial)
        #self.encoder_layers_level = _get_clones(encoder_layer_level, self.num_encoder_layers_level)
        self.decoder_layers = _get_clones(decoder_layer, num_decoder_layers)

        self.spatial_prior=spatial_prior

        #self.level_embed = nn.Embedding(num_feature_levels, d_model)
        self.num_pattern = num_query_pattern
        self.pattern = nn.Embedding(self.num_pattern, d_model)

        self.num_position = num_query_position
        if self.spatial_prior == "learned":
            self.position = nn.Embedding(self.num_position, 2)

        self.adapt_pos2d = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.adapt_pos1d = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.num_layers = num_decoder_layers
        num_classes = 2

        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

        self.refine_box = False

        self._reset_parameters()

    def _reset_parameters(self):

        num_pred = self.num_layers
        num_classes = 2
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if self.spatial_prior == "learned":
            nn.init.uniform_(self.position.weight.data, 0, 1)

        if self.refine_box:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

    def self_attention_anchor(self, x, masks):
        bs, l, c, h, w = x.shape
        mask = masks.unsqueeze(1).repeat(1, l, 1, 1).reshape(bs * l, h, w)
        pos_col, pos_row = mask2pos(mask)
        if self.attention_type == "RCDA":
            posemb_row = self.adapt_pos1d(pos2posemb1d(pos_row))
            posemb_col = self.adapt_pos1d(pos2posemb1d(pos_col))
            posemb_2d = None
        else:
            pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, h, 1).unsqueeze(-1),
                                pos_col.unsqueeze(2).repeat(1, 1, w).unsqueeze(-1)], dim=-1)
            posemb_2d = self.adapt_pos2d(pos2posemb2d(pos_2d))
            posemb_row = posemb_col = None
        outputs = x.reshape(bs * l, c, h, w)

        for idx in range(len(self.encoder_layers)):
            outputs = self.encoder_layers[idx](outputs, mask, posemb_row, posemb_col, posemb_2d)
            if idx < self.num_encoder_layers_level:
                outputs = self.encoder_layers_level[idx](outputs,
                                                              level_emb=self.level_embed.weight.unsqueeze(1).unsqueeze(
                                                                  0).repeat(bs, 1, 1, 1).reshape(bs * l, 1, c))

        return outputs.reshape(bs, l, c, h, w)

    #def forward(self, srcs, masks):
    def forward(self, srcs_temp, masks_temp, srcs_search, masks_search):

        bs, l, c, h, w = srcs_temp.shape
        mask_temp = masks_temp.unsqueeze(1).repeat(1, l, 1, 1).reshape(bs * l, h, w)
        pos_col, pos_row = mask2pos(mask_temp)
        if self.attention_type == "RCDA":
            posemb_row = self.adapt_pos1d(pos2posemb1d(pos_row))
            posemb_col = self.adapt_pos1d(pos2posemb1d(pos_col))
            posemb_2d = None
        else:
            pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, h, 1).unsqueeze(-1),
                                pos_col.unsqueeze(2).repeat(1, 1, w).unsqueeze(-1)], dim=-1)
            posemb_2d = self.adapt_pos2d(pos2posemb2d(pos_2d))
            posemb_row = posemb_col = None
        ##############
        
        bs_, l_, c_, h_, w_ = srcs_search.shape
        mask_temp_ = masks_search.unsqueeze(1).repeat(1, l_, 1, 1).reshape(bs_ * l_, h_, w_)
        pos_col_, pos_row_ = mask2pos(mask_temp_)
        if self.attention_type == "RCDA":
            posemb_row_ = self.adapt_pos1d(pos2posemb1d(pos_row_))
            posemb_col_ = self.adapt_pos1d(pos2posemb1d(pos_col_))
            posemb_2d_ = None
        else:
            pos_2d_ = torch.cat([pos_row_.unsqueeze(1).repeat(1, h_, 1).unsqueeze(-1),
                                pos_col_.unsqueeze(2).repeat(1, 1, w_).unsqueeze(-1)], dim=-1)
            posemb_2d_ = self.adapt_pos2d(pos2posemb2d(pos_2d_))
            posemb_row_ = posemb_col_ = None
        
        ##########

        if self.spatial_prior == "learned":
            reference_points = self.position.weight.unsqueeze(0).repeat(bs, self.num_pattern, 1)
        elif self.spatial_prior == "grid":
            nx=ny=round(math.sqrt(self.num_position))
            self.num_position=nx*ny
            x = (torch.arange(nx) + 0.5) / nx
            y = (torch.arange(ny) + 0.5) / ny
            xy=torch.meshgrid(x,y)
            reference_points=torch.cat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1).cuda()
            reference_points = reference_points.unsqueeze(0).repeat(bs, self.num_pattern, 1)
        else:
            raise ValueError(f'unknown {self.spatial_prior} spatial prior')
        srcs = self.self_attention_anchor(srcs_temp, masks_temp)
        
        #output = self.self_attention_anchor(srcs_search, masks_search)
        output = srcs_search
        
        tgt_b, tgt_l, tag_c, tgt_h, tgt_w = output.shape
        output = output.view(tgt_b, tgt_l, tag_c, tgt_h*tgt_w)
        posemb_2d_ = posemb_2d_.view(tgt_b, tgt_l, tag_c, tgt_h*tgt_w)
        output = output.transpose(2,3).squeeze(1)
        posemb_2d_ = posemb_2d_.transpose(2,3).squeeze(1)

        outputs_classes = []
        outputs_coords = []
        for lid, layer in enumerate(self.decoder_layers):
            output = layer(output, reference_points, srcs, mask_temp, adapt_pos2d=self.adapt_pos2d,
                           adapt_pos1d=self.adapt_pos1d, posemb_row=posemb_row, posemb_col=posemb_col,posemb_2d=posemb_2d,posemb_2d_=posemb_2d_)
            reference = inverse_sigmoid(reference_points)
            outputs_class = self.class_embed[lid](output)
            tmp = self.bbox_embed[lid](output)
            '''if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference'''
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class[None,])
            outputs_coords.append(outputs_coord[None,])
            if self.refine_box:
                reference_points = outputs_coord

        output = torch.cat(outputs_classes, dim=0), torch.cat(outputs_coords, dim=0)

        return output


class TransformerEncoderLayerSpatial(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0., activation="relu",
                 n_heads=8, attention_type="RCDA"):
        super().__init__()

        self.attention_type = attention_type
        if attention_type=="RCDA":
            attention_module=MultiheadRCDA
        elif attention_type == "nn.MultiheadAttention":
            attention_module=nn.MultiheadAttention
        else:
            raise ValueError(f'unknown {attention_type} attention_type')

        # self attention
        self.self_attn = attention_module(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, padding_mask=None, posemb_row=None, posemb_col=None,posemb_2d=None):
        # self attention
        bz, c, h, w = src.shape
        src = src.permute(0, 2, 3, 1)

        if self.attention_type=="RCDA":
            posemb_row = posemb_row.unsqueeze(1).repeat(1, h, 1, 1)
            posemb_col = posemb_col.unsqueeze(2).repeat(1, 1, w, 1)
            src2 = self.self_attn((src + posemb_row).reshape(bz, h * w, c), (src + posemb_col).reshape(bz, h * w, c),
                                  src + posemb_row, src + posemb_col,
                                  src, key_padding_mask=padding_mask)[0].transpose(0, 1).reshape(bz, h, w, c)
        else:
            src2 = self.self_attn((src + posemb_2d).reshape(bz, h * w, c).transpose(0, 1),
                                  (src + posemb_2d).reshape(bz, h * w, c).transpose(0, 1),
                                  src.reshape(bz, h * w, c).transpose(0, 1))[0].transpose(0, 1).reshape(bz, h, w, c)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.ffn(src)
        src = src.permute(0, 3, 1, 2)
        return src


class TransformerEncoderLayerLevel(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0., activation="relu",
                 n_heads=8):
        super().__init__()

        # self attention
        self.self_attn_level = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, level_emb=0):
        # self attention
        bz, c, h, w = src.shape
        src = src.permute(0, 2, 3, 1)

        src2 = self.self_attn_level(src.reshape(bz, h * w, c) + level_emb, src.reshape(bz, h * w, c) + level_emb,
                                    src.reshape(bz, h * w, c))[0].reshape(bz, h, w, c)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.ffn(src)
        src = src.permute(0, 3, 1, 2)
        return src



class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0., activation="relu", n_heads=8,
                 n_levels=3, attention_type="RCDA"):
        super().__init__()

        self.attention_type = attention_type
        self.attention_type = attention_type
        if attention_type=="RCDA":
            attention_module=MultiheadRCDA
        elif attention_type == "nn.MultiheadAttention":
            attention_module=nn.MultiheadAttention
        else:
            raise ValueError(f'unknown {attention_type} attention_type')

        # cross attention
        #self.cross_attn = attention_module(d_model,n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # self attention
        #self.self_attn_level = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        #self.level_fc = nn.Linear(d_model * n_levels, d_model)
        # ffn
        self.ffn = FFN(d_model, d_ffn, dropout, activation)
        self.linears = _get_clones(nn.Linear(d_model, d_model),4)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, srcs, src_padding_masks=None, adapt_pos2d=None,
                adapt_pos1d=None, posemb_row=None, posemb_col=None, posemb_2d=None, posemb_2d_=None):
        '''tgt_len = tgt.shape[1]
        query_pos = pos2posemb2d(reference_points.squeeze(2))
        query_pos = adapt_pos2d(query_pos)'''
        # self attention
        q = k = self.with_pos_embed(tgt, posemb_2d_)
        #q = k = tgt
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        bz, l, c, h, w = srcs.shape
        srcs = srcs.reshape(bz * l, c, h, w).permute(0, 2, 3, 1)

        '''if self.attention_type == "RCDA":
            query_pos_x = adapt_pos1d(pos2posemb1d(reference_points[..., 0]))
            query_pos_y = adapt_pos1d(pos2posemb1d(reference_points[..., 1]))
            posemb_row = posemb_row.unsqueeze(1).repeat(1, h, 1, 1)
            posemb_col = posemb_col.unsqueeze(2).repeat(1, 1, w, 1)
            src_row = src_col = srcs
            k_row = src_row + posemb_row
            k_col = src_col + posemb_col
            tgt2 = self.cross_attn((tgt + query_pos_x).repeat(l, 1, 1), (tgt + query_pos_y).repeat(l, 1, 1), k_row, k_col,
                                   srcs, key_padding_mask=src_padding_masks)[0].transpose(0, 1)
        else:
            tgt2 = self.cross_attn((tgt).repeat(l, 1, 1).transpose(0, 1),
                                   (srcs).reshape(bz * l, h * w, c).transpose(0,1),
                                   srcs.reshape(bz * l, h * w, c).transpose(0, 1))[0].transpose(0,1)

        if l > 1:
            tgt2 = self.level_fc(tgt2.reshape(bz, l, tgt_len, c).permute(0, 2, 3, 1).reshape(bz, tgt_len, c * l))'''
        
        srcs = srcs.reshape(bz * l, h * w, c)
        query, key, value = [l(x) for l, x in zip(self.linears, (tgt, srcs, srcs))]
        #query, key, value = [l(x) for l, x in zip(self.linears, ((tgt + query_pos), (srcs + posemb_2d).reshape(bz * l, h * w, c), srcs.reshape(bz * l, h * w, c)))]
        #affinity = torch.bmm(tgt + query_pos, (srcs + posemb_2d).reshape(bz * l, h * w, c).transpose(1,2))#/math.sqrt(c)###(b,HW,c)(b,c,hw)HW hw
        affinity = torch.bmm(query, key.transpose(1,2))#/math.sqrt(c)
        affinity = torch.softmax(affinity, dim=-1)
        if False:
            affinity = softmax_w_g_top(affinity, top=5, gauss=None)
        tgt2 = torch.bmm(affinity, value)
        tgt2 = self.linears[-1](tgt2)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        return tgt


def softmax_w_g_top(x, top=None, gauss=None):#x(b,HW,hw)
    if top is not None:
        if gauss is not None:
            maxes = torch.max(x, dim=1, keepdim=True)[0]
            x_exp = torch.exp(x - maxes)*gauss
            x_exp, indices = torch.topk(x_exp, k=top, dim=1)
        else:
            values, indices = torch.topk(x, k=top, dim=-1)
            #print(values.shape)8 1024 5
            #print(values[:,0].shape) 8 5
            #print([:,:,0,None].shape)
            x_exp = torch.exp(values - values[:,:,0,None].repeat(1,1,5))
            #x_exp = torch.exp(values - values[:,0])
        x_exp_sum = torch.sum(x_exp, dim=-1, keepdim=True)
        x_exp_ =x_exp / x_exp_sum
        # The types should be the same already
        # some people report an error here so an additional guard is added
        x.zero_().scatter_(-1, indices, x_exp_.type(x.dtype)) # B * THW * HW
        output = x
    else:
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        if gauss is not None:
            x_exp = torch.exp(x-maxes)*gauss

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        output = x_exp

    return output


class FFN(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_featurefusion_network(settings):
    return FeatureFusionNetwork(
        d_model=settings.hidden_dim,
        dropout=settings.dropout,
        nhead=settings.nheads,
        dim_feedforward=settings.dim_feedforward
    )





def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., None] / dim_t
    posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb


def mask2pos(mask):
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
    x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    return y_embed, x_embed


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)
    



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.bmm(p_attn, value), p_attn
