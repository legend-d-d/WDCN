import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCA(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.99, lambda1=1.0, lambda2=1.0):
        super().__init__()
        self.momentum = momentum
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.register_buffer("proto_s", torch.zeros(num_classes, feat_dim))
        self.register_buffer("proto_t", torch.zeros(num_classes, feat_dim))
        self.register_buffer("seen_s", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("seen_t", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def _update(self, features, labels, prototypes, seen):
        for label in labels.unique():
            c = int(label.item())
            mean = features[labels == c].mean(dim=0)
            prototypes[c] = self.momentum * prototypes[c] + (1.0 - self.momentum) * mean
            seen[c] = True

    def forward(self, features_s, labels_s, features_t, logits_t):
        probabilities = F.softmax(logits_t, dim=-1)
        pseudo_labels = probabilities.detach().argmax(dim=1)
        features_s = F.normalize(features_s, dim=-1)
        features_t = F.normalize(features_t, dim=-1)

        self._update(features_s, labels_s, self.proto_s, self.seen_s)
        self._update(features_t, pseudo_labels, self.proto_t, self.seen_t)

        valid = self.seen_s[pseudo_labels] & self.seen_t[pseudo_labels]
        if valid.any():
            features = F.normalize(features_t[valid], dim=-1)
            source = F.normalize(self.proto_s[pseudo_labels[valid]], dim=-1)
            target = F.normalize(self.proto_t[pseudo_labels[valid]], dim=-1)
            distance_s = 1.0 - (features * source).sum(dim=-1)
            distance_t = 1.0 - (features * target).sum(dim=-1)
            alignment = (self.lambda1 * distance_s + self.lambda2 * distance_t).mean()
        else:
            alignment = features_t.sum() * 0.0

        entropy = -(probabilities * (probabilities + 1e-6).log()).sum(dim=1)
        weights = 1.0 + torch.exp(-entropy)
        weights = probabilities.size(0) * weights / weights.sum()
        confusion = (probabilities * weights[:, None]).t() @ probabilities
        confusion = confusion / confusion.sum(dim=1, keepdim=True)
        confusion_loss = (confusion.sum() - confusion.trace()) / confusion.size(0)
        return alignment, confusion_loss


def adaptation_weight(epoch, max_epochs, zeta=8.0):
    return 2.0 / (1.0 + math.exp(-zeta * (epoch - 1) / (max_epochs - 1))) - 1.0
