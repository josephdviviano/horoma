import torch.nn as nn
import Preprocessor as pp
import math
from transformers import EncoderLayer, EncoderTaskLayer, EncoderTaskLayer2

class TransformerNet(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 n_layers=2,
                 kernel_size=2,
                 pool_size=2,
                 n_heads=4,
                 key_dim=None,
                 val_dim=None,
                 inner_dim=None,
                 dropout=0.1
                 ):
        super(TransformerNet, self).__init__()
        # self.preprocess = pp.Preprocessor()

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        # size of output
        lout = self.l_out_conv1d(3072, kernel_size)
        self.pool1 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1d(lout, pool_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1d(lout, kernel_size)
        self.pool2 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1d(lout, pool_size)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1d(lout, kernel_size)
        self.pool3 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1d(lout, pool_size)

        print('lout: ', lout)

        self.nl = nn.ReLU()

        if key_dim is None:
            key_dim = hidden_size // n_heads

        if val_dim is None:
            val_dim = hidden_size // n_heads

        if inner_dim is None:
            inner_dim = hidden_size // 2

        self.layer_stack = [] if n_layers == 0 else nn.ModuleList([
            EncoderLayer(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        if not isinstance(output_size, (list, tuple)):
            output_size = [output_size]

        output_modules = [
            nn.Sequential(
                # EncoderLayer(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout, attn_flag=False),
                # EncoderLayer(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout, attn_flag=False),
                EncoderTaskLayer2(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout, attn_flag=False),
                nn.Linear(hidden_size, 200),
                nn.ReLU(),
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Linear(200, o)
            ) for o in output_size
        ]
        if len(output_modules) == 1:
            self.out = output_modules[0]
        else:
            self.out = nn.ModuleList(output_modules)

    def l_out_conv1d(self, l_in, kernel_size, stride=1, padding=0, dilation=1):
        l_out = (l_in + (2 * padding) - dilation *
                 (kernel_size - 1) - 1) / stride
        l_out = l_out + 1
        return int(l_out)

    def l_out_maxpool1d(self, l_in, kernel_size, stride=None, padding=0, dilation=1):
        if stride is None:
            stride = kernel_size
        l_out = self.l_out_conv1d(
            l_in, kernel_size, stride, padding, dilation
        )
        return l_out

    def forward(self, x, noise=None):
        # x = self.preprocess(x)
        if noise is not None:
            x = x + noise

        x = self.nl(self.pool1(self.conv1(x)))
        x = self.nl(self.pool2(self.conv2(x)))
        x = self.nl(self.pool3(self.conv3(x)))

        data = x.permute(0, 2, 1)

        for enc_layer in self.layer_stack:
            data, _ = enc_layer(data)

        if isinstance(self.out, nn.ModuleList):
            pred = [
                l(data) for i, l in enumerate(self.out)
            ]
        else:
            pred = self.out(data)

        return pred


class TransformerNet2D(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 n_layers=2,
                 kernel_size=2,
                 pool_size=2,
                 n_heads=4,
                 key_dim=None,
                 val_dim=None,
                 inner_dim=None,
                 dropout=0.2,
                 use_transformer=True
                 ):
        super(TransformerNet2D, self).__init__()
        # self.preprocess = pp.Preprocessor()

        self.use_transformer = use_transformer

        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size)
        # size of output
        lout = self.l_out_conv2d(32, kernel_size)
        self.pool1 = nn.MaxPool2d(pool_size)
        lout = self.l_out_maxpool2d(lout, pool_size)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv2d(lout, kernel_size)
        self.pool2 = nn.MaxPool2d(pool_size)
        lout = self.l_out_maxpool2d(lout, pool_size)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv2d(lout, kernel_size)
        self.pool3 = nn.MaxPool2d(pool_size)
        lout = self.l_out_maxpool2d(lout, pool_size)

        print('lout: ', lout)
        self.lout = lout

        self.nl = nn.ReLU()

        if key_dim is None:
            key_dim = hidden_size // n_heads

        if val_dim is None:
            val_dim = hidden_size // n_heads

        if inner_dim is None:
            inner_dim = hidden_size // 2

        self.layer_stack = [] if n_layers == 0 else nn.ModuleList([
            EncoderLayer(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        if not isinstance(output_size, (list, tuple)):
            output_size = [output_size]

        output_modules = [
            nn.Sequential(
                # EncoderLayer(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout, attn_flag=False),
                # EncoderLayer(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout, attn_flag=False),
                EncoderTaskLayer2(hidden_size, inner_dim, n_heads, key_dim, val_dim, dropout=dropout, attn_flag=False),
                nn.Linear(hidden_size, 200),
                nn.ReLU(),
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Linear(200, o)
            ) for o in output_size
        ]
        if not self.use_transformer:
            print("Not using transformer")
            output_modules = [
                nn.Sequential(
                    nn.Linear(lout*lout*hidden_size, 200),
                    nn.ReLU(),
                    nn.Linear(200, 200),
                    nn.ReLU(),
                    nn.Linear(200, o)
                ) for o in output_size
            ]

        if len(output_modules) == 1:
            self.out = output_modules[0]
        else:
            self.out = nn.ModuleList(output_modules)

    def l_out_conv2d(self, l_in, kernel_size, stride=1, padding=0, dilation=1):
        l_out = (l_in + (2 * padding) - dilation *
                 (kernel_size - 1) - 1) / stride
        l_out = l_out + 1
        return int(l_out)

    def l_out_maxpool2d(self, l_in, kernel_size, stride=None, padding=0, dilation=1):
        if stride is None:
            stride = kernel_size
        l_out = self.l_out_conv2d(
            l_in, kernel_size, stride, padding, dilation
        )
        return l_out

    def forward(self, x, noise=None):
        # x = self.preprocess(x)
        if noise is not None:
            x = x + noise

        x = self.nl(self.pool1(self.conv1(x)))
        x = self.nl(self.pool2(self.conv2(x)))
        x = self.nl(self.pool3(self.conv3(x)))

        if self.use_transformer:
            x = x.view(x.shape[0], x.shape[1], self.lout*self.lout)
            data = x.permute(0, 2, 1)
        else:
            data = x.view(x.shape[0], -1)

        for enc_layer in self.layer_stack:
            data, _ = enc_layer(data)

        if isinstance(self.out, nn.ModuleList):
            pred = [
                l(data) for i, l in enumerate(self.out)
            ]
        else:
            pred = self.out(data)

        return pred