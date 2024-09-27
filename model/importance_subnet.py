from torch import nn
from dac_repo.dac.nn.layers import Snake1d, WNConv1d


class ImportanceSubnet(nn.Module):
    def __init__(
        self,
        d_feat, # d_feat = 1024
        detach_input: bool = False,
    ):
        super().__init__()
        self._init_weights_zero()
        
        self.block1 = nn.Sequential(
            Snake1d(d_feat),
            WNConv1d(d_feat, d_feat//2, kernel_size=5, padding=2),
        )
        self.block2 = nn.Sequential(
            Snake1d(d_feat//2),
            WNConv1d(d_feat//2, d_feat//8, kernel_size=3, padding=1),
            Snake1d(d_feat//8),
        )
        self.block3 = nn.Sequential(
            Snake1d(d_feat//8),
            WNConv1d(d_feat//8, d_feat//32, kernel_size=3, padding=1),
        )
        self.block4 = nn.Sequential(
            Snake1d(d_feat//32),
            WNConv1d(d_feat//32, d_feat//128, kernel_size=3, padding=1),
        )
        self.block5 = nn.Sequential(
            Snake1d(d_feat//128),
            WNConv1d(d_feat//128, 1, kernel_size=1),
        )
        self.act_fn = nn.Sigmoid()
        self.detach_input = detach_input

    def forward(self, x_in):
        if self.detach_input:
            x_in = x_in.detach()
        x_1 = self.block1(x_in) ## x1: (B, 512, T)
        x_2 = self.block2(x_1)
        x_3 = self.block3(x_2) ## x3: (B, 128, T)
        x_4 = self.block4(x_3)
        x_5 = self.block5(x_4) ## x5: (B, 1, T)
        out = self.act_fn(x_5)

        return out # (B, 1, 32)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _init_weights_zero(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.zeros_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)