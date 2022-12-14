import pretrainedmodels
import torch.nn as nn

from .imagenet_ensemble import ImagenetEnsemble
from codes.model.utils import util_vit


def load_imagenet_model(model_type):
    if model_type == 'ensemble':
        model = ImagenetEnsemble()
    else:
        model = pretrainedmodels.__dict__[model_type](
            num_classes=1000, pretrained='imagenet').eval()
        for param in model.parameters():
            param.requires_grad = False
    # model.eval()
    return model

def load_vit_model(args):
    (tar_model, _), tar_mean, tar_std = util_vit.get_model(args.tar_model, args)
    tar_model = tar_model.to(args.device)
    tar_model.eval()
    return tar_model,tar_mean,tar_std