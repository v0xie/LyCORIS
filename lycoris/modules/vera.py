import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import ModuleCustomSD

#SHARED_A = None
#SHARED_B = None

class VeRAModule(ModuleCustomSD):
    """
    modifed from glora.py
    implementation of "arXiv:2310.11454 [cs.CL]" https://arxiv.org/abs/2310.11454
    """

    def __init__(
        self, 
        lora_name, org_module: nn.Module, 
        multiplier=1.0, 
        lora_dim=8, alpha=1, 
        dropout=0., rank_dropout=0., module_dropout=0.,
        rank_dropout_scale=False,
        *args,
        **kwargs,
    ):
        #global SHARED_A, SHARED_B

        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.cp = False


        #if args.seed is None:
        #    args.seed = 1
        self.seed = 1

        if isinstance(org_module, nn.Linear):
            self.op = F.linear
            self.in_dim = org_module.in_features
            self.out_dim = org_module.out_features
            self.kw_dict = {}
        elif isinstance(org_module, nn.Conv2d):
            self.op = F.conv2d
            self.in_dim = org_module.in_channels
            self.out_dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        else:
            raise NotImplementedError

        self.shape = org_module.weight.shape
        if isinstance(org_module, nn.Conv2d):
            #assert org_module.kernel_size == (1,1)
            self.is_conv = True
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            k_size = org_module.kernel_size
            self.shared_b = torch.empty([self.shape[0], self.lora_dim])
            self.shared_a = torch.empty((self.lora_dim, self.shape[1]))
            self.b = nn.Conv2d(1, self.shape[0], k_size, bias=False)
            self.d = nn.Conv2d(1, self.lora_dim, k_size, bias=False)
        elif isinstance(org_module, nn.Linear):
            self.is_conv = False
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.shared_b = torch.empty([self.shape[0], self.lora_dim])
            self.shared_a = torch.empty([self.lora_dim, self.shape[1]])
            self.b = nn.Linear(1, self.shape[0], bias=False)
            self.d = nn.Linear(1, self.lora_dim, bias=False)
        else:
            self.is_conv = False
            raise NotImplementedError

        torch.manual_seed(self.seed+1)
        torch.nn.init.kaiming_uniform_(self.shared_b, a=math.sqrt(5))
        self.register_buffer('mat_b', self.shared_b)

        torch.manual_seed(self.seed)
        torch.nn.init.kaiming_uniform_(self.shared_a, a=math.sqrt(5))
        self.register_buffer('mat_a', self.shared_a)

        d_init = 10.0e-07 # ablation pg.9
        torch.nn.init.constant_(self.b.weight, 0)
        torch.nn.init.constant_(self.d.weight, d_init)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout
        
        #if type(alpha) == torch.Tensor:
        #    alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha)) # 定数として扱える

        # same as microsoft's
        #torch.nn.init.kaiming_uniform_(self.a1.weight, a=math.sqrt(5))
        #torch.nn.init.zeros_(self.a2.weight)
        #torch.nn.init.kaiming_uniform_(self.b1.weight, a=math.sqrt(5))
        #torch.nn.init.zeros_(self.b2.weight)

        self.multiplier = multiplier
        self.org_module = [org_module]

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
    
    def get_a(self):
        torch.manual_seed(self.seed+1)
        torch.nn.init.kaiming_uniform_(self.shared_b, a=math.sqrt(5))
        self.register_buffer('mat_b', self.shared_b)

    def get_b(self):
        shared_b = torch.empty([self.shape[0], self.lora_dim])
        torch.manual_seed(self.seed+1)

        torch.nn.init.kaiming_uniform_(self.shared_b, a=math.sqrt(5))
        self.register_buffer('mat_b', self.shared_b)
    
    def make_weight(self, device=None):
        b = self.b.weight.view(self.b.weight.size(0), -1)
        d = self.d.weight.view(self.d.weight.size(0), -1)
        wb = torch.diag(b.flatten()).to(device=device)
        wd = torch.diag(d.flatten()).to(device=device)
        return wb @ self.mat_b @ wd @ self.mat_a

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.scale * self.multiplier

        w = self.make_weight(device=x.device)

        if self.is_conv:
            w = w.view(-1, w.size(1), 1, 1)

        kw_dict = self.kw_dict | {"weight": w, "bias": self.org_module[0].bias}
        return self.org_forward(x) + self.op(x, **kw_dict)
