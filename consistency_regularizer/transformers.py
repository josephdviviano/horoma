"""
this script is taken as is from b2pomt_baseline
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # print('q: ', q.size())
        # print('k: ', k.size())
        # print('v: ', v.size())

        # print('n_head: ', self.n_head)
        # print('d_k: ', self.d_k)
        # print('d_v: ', self.d_v)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class MultiHeadTaskAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.num_embeddings = 1
        self.d_model = d_model
        self.embedding = nn.Embedding(self.num_embeddings, d_model)
        self.multihead = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)

    def forward(self, q, k, v, mask=None):
        assert type(q) == int
        assert q < self.num_embeddings
        # q = torch.LongTensor([[q]]).expand(k.size(0), 1).to(k.device)
        # q = self.embedding(q)
        q = (torch.ones((k.size(0), 1, self.d_model)) / self.d_model).to(k.device)

        return self.multihead(q, k, v, mask)


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1, d_out=None):
        super().__init__()
        if d_out is None:
            d_out = d_in
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_out, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.d_out = d_out
        self.d_in = d_in

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        if self.d_out == self.d_in:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output)
        return output


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        d_out=None,
        attn_flag=True,
    ):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, d_out=d_out
        )
        self.attn_flag = attn_flag

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        if self.attn_flag:
            return enc_output, enc_slf_attn
        else:
            return enc_output


class EncoderTaskLayer(nn.Module):
    """ Compose with two layers """

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        d_out=None,
        attn_flag=True,
    ):
        super(EncoderTaskLayer, self).__init__()
        self.slf_attn = MultiHeadTaskAttention(
            n_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, d_out=d_out
        )
        self.attn_flag = attn_flag

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            0, enc_input, enc_input, mask=slf_attn_mask
        )

        enc_output = self.pos_ffn(enc_output)

        if self.attn_flag:
            return enc_output.squeeze(1), enc_slf_attn
        else:
            return enc_output.squeeze(1)


class EncoderTaskLayer2(nn.Module):
    """ Compose with two layers """

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        d_out=None,
        attn_flag=True,
    ):
        super(EncoderTaskLayer2, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, d_out=d_out
        )
        self.attn_flag = attn_flag

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )

        enc_output = self.pos_ffn(enc_output)

        enc_output = enc_output.sum(dim=1, keepdim=True)

        if self.attn_flag:
            return enc_output.squeeze(1), enc_slf_attn
        else:
            return enc_output.squeeze(1)
