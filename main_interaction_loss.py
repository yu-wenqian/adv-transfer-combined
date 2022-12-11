import argparse

from codes.basic_functions.transferability import (interaction_reduced_attack,
                                                   leave_one_out)
from set_config import set_config

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--p", type=str, help="inf; 2", default='inf')
parser.add_argument("--epsilon", type=int, default=16)
parser.add_argument("--step_size", type=float, default=2.)
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--loss_root", type=str, default='./experiments/loss')
parser.add_argument(
    "--adv_image_root", type=str, default='./experiments/adv_images')
parser.add_argument("--clean_image_root", type=str, default='data/images_1000')
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--src_model", type=str, default='resnet50')
parser.add_argument("--src_kind", type=str, default='cnn')
parser.add_argument(
    "--tar_model", type=str, default=['vit'], nargs='*')
parser.add_argument("--tar_kind", type=str, default='cnn')

# args for different attack methods
parser.add_argument("--attack_method", type=str, default='PGD+VI')

parser.add_argument("--gamma", type=float, default=1.)
parser.add_argument("--momentum", type=float, default=1.)
parser.add_argument("--m", type=int, default=0)
parser.add_argument("--sigma", type=float, default=15.)
parser.add_argument("--ti_size", type=int, default=1)
parser.add_argument("--lam", type=float, default=0.)
parser.add_argument("--grid_scale", type=int, default=16)
parser.add_argument("--sample_grid_num", type=int, default=32)
parser.add_argument("--sample_times", type=int, default=32)
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--image_resize", type=int, default=255)
parser.add_argument("--prob", type=float, default=0.4)
parser.add_argument('--linbp_layer', type=str, default='3_1')
parser.add_argument('--ila_layer', type=str, default='2_3')
parser.add_argument('--ila_niters', type=int, default=100)
#args for VI
parser.add_argument("--beta", type=float, default=1.5)
parser.add_argument('--number', type=int, default=20)

# args for vit models
# ghost vit/deit
parser.add_argument('--skip_eros', default=0.01, type=float)
parser.add_argument('--drop_layer', default=0.01, type=float)
# tokenvit
parser.add_argument('--token_combi', default='', type=str)

# parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--device', default='cpu', type=str)

args = parser.parse_args()

# tar_model_list = [
#     "vgg16", "resnet152", "densenet201", "senet154", "inceptionv3",
#     "inceptionv4", "inceptionresnetv2"
# ]
tar_model_list = [
    # "vit_base_patch16_224",
    "vgg16",
]

def ttest_interaction_reduced_attack():
    set_config(args)
    interaction_reduced_attack.generate_adv_images(args)
    for tar_model in tar_model_list:
        args.tar_model = tar_model
        interaction_reduced_attack.save_scores(args)
        leave_one_out.evaluate(args)
    # args.tar_model = 'vit'
    # interaction_reduced_attack.save_scores(args)
    # leave_one_out.evaluate(args)


ttest_interaction_reduced_attack()
