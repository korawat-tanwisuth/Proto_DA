from typing import Optional
import torch
import torch.nn as nn
from dalib.modules.classifier import Classifier as ClassifierBase
import torch.nn.functional as F

__all__ = ['ProtoLoss']


class ProtoLoss(nn.Module):
    """
    Parameters:
        - **nav_t** (float): temperature parameter (1 for all experiments)
        - **beta** (float): learning rate/momentum update parameter for learning target proportions
        - **num_classes** (int): total number of classes
        - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

    Inputs: mu_s, f_t
        - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

    """

    def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
        super(ProtoLoss, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.prop = (torch.ones((num_classes,1))*(1/num_classes)).to(device)
        self.eps = 1e-6
         
    def pairwise_cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1 - torch.matmul(x, y.T)

    def get_pos_logits(self, sim_mat, prop):
        log_prior = torch.log(prop + self.eps)
        return sim_mat/self.nav_t + log_prior

    def update_prop(self, prop):
        return (1 - self.beta) * self.prop + self.beta * prop 

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        # Update proportions
        sim_mat = torch.matmul(mu_s, f_t.T)
        old_logits = self.get_pos_logits(sim_mat.detach(), self.prop)
        s_dist_old = F.softmax(old_logits, dim=0)
        prop = s_dist_old.mean(1, keepdim=True)
        self.prop = self.update_prop(prop)

        # Calculate bi-directional transport loss
        new_logits = self.get_pos_logits(sim_mat, self.prop)
        s_dist = F.softmax(new_logits, dim=0)
        t_dist = F.softmax(sim_mat/self.nav_t, dim=1)
        cost_mat = self.pairwise_cosine_dist(mu_s, f_t)
        source_loss = (self.s_par*cost_mat*s_dist).sum(0).mean() 
        target_loss = (((1-self.s_par)*cost_mat*t_dist).sum(1)*self.prop.squeeze(1)).sum()
        loss = source_loss + target_loss
        return loss


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)
