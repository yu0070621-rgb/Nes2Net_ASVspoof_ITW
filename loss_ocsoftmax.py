import torch
import torch.nn as nn
import torch.nn.functional as F

class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=1024, r_real=0.5, r_fake=0.2, alpha=10.0):#r_real:0.9->0.5,alpha:20.0->10.0
        super().__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real    # bonafide margin
        self.r_fake = r_fake    # spoof margin
        self.alpha = alpha      # scale
        self.weight = nn.Parameter(torch.FloatTensor(feat_dim, 1))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, labels):
        """
        x: (B, feat_dim) embedding before FC
        labels: (B,) 0=bonafide, 1=spoof
        """
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=0)
        cos_theta = torch.matmul(x_norm, w_norm).squeeze(-1)  # (B,)

        # bonafide: cos_theta should be close to 1
        # spoof:    cos_theta should be close to -1
        cos_theta_bonafide = cos_theta * (labels == 0).float()
        cos_theta_spoof    = cos_theta * (labels == 1).float()

        loss = torch.log(1 + torch.exp(self.alpha * (self.r_real - cos_theta_bonafide))) \
             + torch.log(1 + torch.exp(self.alpha * (cos_theta_spoof - self.r_fake)))
        return loss.mean()