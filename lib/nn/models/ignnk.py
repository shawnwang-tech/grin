import torch
from einops import rearrange
from torch import nn

from ..layers import IGNNK_model
from ...utils.parser_utils import str_to_bool
from ..utils.ops import calculate_random_walk_matrix


class IGNNK(nn.Module):
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 d_ff,
                 ff_dropout,
                 n_layers=1,
                 kernel_size=2,
                 decoder_order=1,
                 global_att=False,
                 d_u=0,
                 d_emb=0,
                 layer_norm=False,
                 merge='mlp',
                 impute_only_holes=True,
                 window=24):
        super(IGNNK, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_u = int(d_u) if d_u is not None else 0
        self.d_emb = int(d_emb) if d_emb is not None else 0
        self.register_buffer('adj', torch.tensor(adj).float())

        A_q = torch.from_numpy((calculate_random_walk_matrix(adj).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(adj).T).astype('float32'))

        self.register_buffer('A_q', torch.tensor(A_q).float())
        self.register_buffer('A_h', torch.tensor(A_h).float())

        self.impute_only_holes = impute_only_holes

        self.ignnk = IGNNK_model(h=window, z=100, k=1)

    def forward(self, x, mask=None, u=None, **kwargs):
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]

        x = x[...,None]
        mask = mask[...,None]
        # For IGNNK
        x[mask == 0] = 0

        x = rearrange(x, 'b s n c -> b c n s')
        if mask is not None:
            mask = rearrange(mask, 'b s n c -> b c n s')

        if u is not None:
            u = rearrange(u, 'b s n c -> b c n s')

        imputation = self.ignnk(x[:,0].transpose(-2, -1), self.A_q, self.A_h)

        # # imputation: [batches, channels, nodes, steps] prediction: [4, batches, channels, nodes, steps]
        # imputation, prediction = self.bigrill(x, self.adj, mask=mask, u=u, cached_support=self.training)
        # # In evaluation stage impute only missing values
        # if self.impute_only_holes and not self.training:
        #     imputation = torch.where(mask, x, imputation)
        # # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        # imputation = torch.transpose(imputation, -3, -1)
        # prediction = torch.transpose(prediction, -3, -1)
        # if self.training:
        #     return imputation, prediction

        return imputation

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-hidden', type=int, default=64)
        parser.add_argument('--d-ff', type=int, default=64)
        parser.add_argument('--ff-dropout', type=int, default=0.)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--kernel-size', type=int, default=2)
        parser.add_argument('--decoder-order', type=int, default=1)
        parser.add_argument('--d-u', type=int, default=0)
        parser.add_argument('--d-emb', type=int, default=8)
        parser.add_argument('--layer-norm', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--global-att', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--merge', type=str, default='mlp')
        parser.add_argument('--impute-only-holes', type=str_to_bool, nargs='?', const=True, default=True)
        parser.add_argument('--window', type=int, default=24)
        return parser
