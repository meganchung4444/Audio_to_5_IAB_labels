import torch
import torch.nn.functional as F


def clip_nll(output_dict, target_dict):
    loss = F.binary_cross_entropy(torch.sigmoid(output_dict['clipwise_output']).to(torch.float32), target_dict['target'].to(torch.float32))
    return loss

def get_loss_func(loss_type):
    if loss_type == 'clip_nll':
        return clip_nll
