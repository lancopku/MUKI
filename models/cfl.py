import torch
import torch.nn as nn
import torch.nn.functional as F


def _mmd_rbf2(x, y, sigmas=None):
    N, _ = x.shape
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = L = P = 0.0
    XX2 = rx.t() + rx - 2 * xx
    YY2 = ry.t() + ry - 2 * yy
    XY2 = rx.t() + ry - 2 * zz

    if sigmas is None:
        sigma2 = torch.mean((XX2.detach() + YY2.detach() + 2 * XY2.detach()) / 4)
        sigmas2 = [sigma2 / 4, sigma2 / 2, sigma2, sigma2 * 2, sigma2 * 4]
        alphas = [1.0 / (2 * sigma2) for sigma2 in sigmas2]
    else:
        alphas = [1.0 / (2 * sigma ** 2) for sigma in sigmas]

    for alpha in alphas:
        K += torch.exp(- alpha * (XX2.clamp(min=1e-12)))
        L += torch.exp(- alpha * (YY2.clamp(min=1e-12)))
        P += torch.exp(- alpha * (XY2.clamp(min=1e-12)))

    beta = (1. / (N * N))
    gamma = (2. / (N * N))

    return F.relu(beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P))


def mmd_loss(f1, f2, sigmas, normalized=False):
    if len(f1.shape) != 2:
        N = f1.size(0)
        f1 = f1.view(N, -1)
        N = f2.size(0)
        f2 = f2.view(N, -1)

    if normalized:
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

    return _mmd_rbf2(f1, f2, sigmas=sigmas)


class CFLLoss(nn.Module):
    """ Common Feature Learning Loss
        CFL Loss = MMD + MSE
    """

    def __init__(self, sigmas, normalized=True):
        super(CFLLoss, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized

    def forward(self, hs, hts, fts_, fts):
        mmd = mse = 0.0
        for ht_i in hts:
            # MMD loss for common feature learning
            mmd += mmd_loss(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized)
        for i in range(len(fts_)):
            # reconstruction on teacher feature loss
            mse += F.mse_loss(fts_[i], fts[i])
        return mmd, mse


class CFLFCBlock(nn.Module):
    """Common Feature Blocks for Fully-Connected layer

    This module is used to capture the common features of multiple teachers and calculate mmd with features of student.
    **Parameters:**
        - cs (int): channel number of student features
        - channel_ts (list or tuple): channel number list of teacher features
        - ch (int): channel number of hidden features
    """

    def __init__(self, config, t_configs, teacher_number=2):
        super(CFLFCBlock, self).__init__()

        self.align_t = nn.ModuleList()
        for i in range(teacher_number):
            self.align_t.append(
                nn.Sequential(
                    nn.Linear(t_configs[i].hidden_size, config.hidden_size // 4),
                    nn.ReLU(inplace=True)
                )
            )

        self.align_s = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(inplace=True),
        )

        self.extractor = nn.Sequential(
            nn.Linear(config.hidden_size // 4, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, config.hidden_size // 4),
        )

        self.dec_t = nn.ModuleList()
        for i in range(teacher_number):
            self.dec_t.append(
                nn.Sequential(
                    nn.Linear(config.hidden_size // 4, t_configs[i].hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(t_configs[i].hidden_size, t_configs[i].hidden_size)
                )
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fs, fts):
        aligned_t = [self.align_t[i](fts[i]) for i in range(len(fts))]
        aligned_s = self.align_s(fs)

        hts = [self.extractor(f) for f in aligned_t]
        hs = self.extractor(aligned_s)

        _fts = [self.dec_t[i](hts[i]) for i in range(len(hts))]
        return (hs, hts), (_fts, fts)
