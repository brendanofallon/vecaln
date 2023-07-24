
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fn(y, aug_y, neg, neg_factor=1.0):
    yn = F.normalize(y, p=2, dim=-1)
    yan = F.normalize(aug_y, p=2, dim=-1)
    ny = F.normalize(neg, p=2, dim=-1)

    lpos = 2 - 2 * (yn * yan).sum(-1)
    lneg = 2 - 2 * (yn * ny).sum(-1)
    lneg2 = 2 - 2 * (yan * ny).sum(-1)

    return lpos - neg_factor * (lneg + lneg2)


class Contrastive(nn.Module):

    def __init__(self, model, transforms, augs):
        super().__init__()
        self.model = model
        self.tr = transforms
        self.augs = augs

    def forward(self, x, negatives, neg_factor=1.0):
        b_size = len(x)
        aug_x = self.augs(x)
        tensor_x = self.tr(x)

        everything = torch.concat([tensor_x, aug_x, negatives], dim=0)
        y = self.model(everything)

        loss = loss_fn(y[0:b_size, :], y[b_size:2*b_size, :], y[2*b_size:, :], neg_factor)
        return loss.mean()



