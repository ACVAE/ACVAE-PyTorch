import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(pred, y_real, cur_cnt, padding, hyper_params):
    pred_out = F.log_softmax(pred, dim=-1)
    padding = (1.0 - padding.float()).unsqueeze(2)
    y_real = F.one_hot(y_real, num_classes=hyper_params['total_items']+1)

    likelihood = -1.0 * \
        torch.sum(pred_out * y_real * padding) / float(pred.shape[0])

    return likelihood


def kl_loss(adversary, x_real, z_inferred, padding, KL_WEIGHT):
    t_joint = adversary(x_real, z_inferred, padding)
    t_joint = torch.mean(t_joint, dim=-1)
    kl = torch.sum(t_joint) / float(x_real.shape[0]) * KL_WEIGHT
    return kl


def adversary_loss(adversary, x_real, z_inferred, padding, CONTRAST_WEIGHT):
    t_joint = adversary(x_real, z_inferred, padding)
    z_shuffled = torch.cat([z_inferred[1:], z_inferred[:1]], dim=0)
    t_shuffled = adversary(x_real, z_shuffled, padding)

    Ej = -F.softplus(-t_joint)
    Em = F.softplus(t_shuffled)
    GLOBAL = (Em - Ej) * (1.0 - padding.float())
    GLOBAL = torch.sum(GLOBAL) / x_real.shape[0] * CONTRAST_WEIGHT

    return GLOBAL


def adversary_kl_loss(adversary_prior, x_real, z_inferred, padding):
    prior = torch.randn_like(z_inferred)
    term_a = torch.log(torch.sigmoid(
        adversary_prior(x_real, z_inferred, padding)) + 1e-9)
    term_b = torch.log(1.0 - torch.sigmoid(adversary_prior(x_real,
                                                           prior, padding)) + 1e-9)
    PRIOR = -torch.mean(term_a + term_b, dim=-1) * (1.0 - padding.float())
    PRIOR = torch.sum(PRIOR) / x_real.shape[0]

    return PRIOR


class MetricShower():
    def __init__(self):
        self.metrics = {}
        self.metrics_cnt = {}

    def store(self, metrics: dict):
        for (k, v) in metrics.items():
            if self.metrics.get(k) is None:
                self.metrics[k] = 0.0
                self.metrics_cnt[k] = 0.0
            self.metrics[k] += v
            self.metrics_cnt[k] += 1

    def get(self, name):
        result = self.metrics[name] / self.metrics_cnt[name]

        return result

    def clear(self):
        for (k, v) in self.metrics.items():
            self.metrics[k] = 0.0
            self.metrics_cnt[k] = 0
