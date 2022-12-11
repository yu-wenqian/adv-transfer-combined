import os

import numpy as np
import torch
import torch.nn as nn
from codes.basic_functions.transferability.get_attacker import get_attacker
from codes.dataset.load_images import load_images
from codes.dataset.save_images import save_images
from codes.model.load_model import load_imagenet_model,load_vit_model
from codes.model.normalizer import Normalize
from codes.utils import reset_dir
from tqdm import tqdm


def generate_adv_images(args):
    print(f'Source arch {args.src_model}')
    print(f'Source kind {args.src_kind}')
    print(f'Attack: {args.attack_method}')
    # print(f'Attack kind: {args.tar_kind}')
    print(
        f'Args: momentum {args.momentum}, gamma {args.gamma}, m {args.m}, sigma {args.sigma}, grid {args.sample_grid_num}, times {args.sample_times}, lam {args.lam}'
    )

    if 'cnn' in args.src_kind:
        model = load_imagenet_model(model_type=args.src_model)
        mean, std = model.mean, model.std
        height, width = model.input_size[1], model.input_size[2]
    elif 'vit' in args.src_kind:
        model, mean, std = load_vit_model(args)
        height = args.image_size
        width = args.image_size

    # height, width = model.input_size[1], model.input_size[2]
    # mean, std = model.mean, model.std
    predict = nn.Sequential(Normalize(mean=mean, std=std), model).to(args.device)

    ori_dataloader, _ = load_images(
        input_dir=args.clean_image_root,
        input_height=height,
        input_width=width)

    adversary = get_attacker(args, predict=predict, image_dim=height * width * 3, image_size=height)

    save_root = args.adv_image_root
    reset_dir(save_root)
    for epoch in range(args.num_steps):
        epoch_save_root = os.path.join(save_root, f'epoch_{epoch}')
        reset_dir(epoch_save_root)
    reset_dir(args.loss_root)
    loss_record = {}

    for (ori_image, label, file_names) in tqdm(ori_dataloader):

        ori_image = ori_image.to(args.device)
        with torch.no_grad():
            out = predict(ori_image)
            pred = out.max(1)[1].item()
            if label.item() != pred:
                raise Exception('Invalid prediction')

        if ('linbp' in args.attack_method or 'ila' in args.attack_method):
            advs, loss_record_i = adversary.perturb_linbp_ila(
                ori_image,
                torch.tensor([label]).to(args.device),
            )
        else:
            advs, loss_record_i = adversary.perturb(
                ori_image,
                torch.tensor([label]).to(args.device),
            )

        advs_len = len(advs)

        advs = advs.detach().cpu().numpy()
        file_name = file_names[0]
        for epoch in range(advs_len):
            epoch_save_root = os.path.join(save_root, f'epoch_{epoch}')
            save_images(
                images=advs[epoch:epoch + 1, :, :, :],
                filename=file_name,
                output_dir=epoch_save_root)
        loss_record[file_name] = loss_record_i

    np.save(os.path.join(args.loss_root, 'loss_record.npy'), loss_record)


def score_function(model, image, label, mode='untarget'):
    if mode == 'untarget':
        with torch.no_grad():
            logits = model(image)
            image_num = len(label)
            scores = np.zeros(image_num)
            preds = np.zeros(image_num)
            for i in range(image_num):
                if isinstance(logits, dict):
                    output = torch.zeros_like(list(logits.values())[0])
                    for arch in logits.keys():
                        output += logits[arch]
                    logits = output
                logits_i = logits[i:i + 1].clone()
                label_i = label[i].item()
                logits_i[:, label_i] = -np.inf
                other_max = logits_i.max(1)[1].item()
                print(logits[i, other_max])
                print(logits[i, label_i])
                scores[i] = (logits[i, other_max] - logits[i, label_i]).item()
                preds[i] = logits[i:i + 1].max(1)[1].item()
            return scores, preds
    else:
        raise NotImplementedError('only implement untarget setting')


def save_scores(args):

    if 'cnn' in args.tar_kind:
        model = load_imagenet_model(model_type=args.src_model)
        mean, std = model.mean, model.std
        height, width = model.input_size[1], model.input_size[2]
    elif 'vit' in args.tar_kind:
        model, mean, std = load_vit_model(args)
        height = args.image_size
        width = args.image_size

    # height, width = model.input_size[1], model.input_size[2]
    # mean, std = model.mean, model.std
    model = nn.Sequential(Normalize(mean=mean, std=std), model).to(args.device)
    adv_scores_dict = {}
    predicts_dict = {}

    for epoch in tqdm(range(args.num_steps)):
        adv_dir = os.path.join(args.adv_image_root, f'epoch_{epoch}')

        adv_dataloader, _ = load_images(
            input_dir=adv_dir,
            input_height=height,
            input_width=width,
            batch_size=128)
        for (adv_images, labels, file_names) in adv_dataloader:
            adv_images = adv_images.to(args.device)
            adv_scores, preds = score_function(model, adv_images, labels)
            for i, fname in enumerate(file_names):
                adv_scores_dict.setdefault(fname, np.zeros(args.num_steps))
                predicts_dict.setdefault(fname, np.zeros(args.num_steps))
                adv_scores_dict[fname][epoch] = adv_scores[i]
                predicts_dict[fname][epoch] = preds[i]
    adv_save_name = f'score_record_{args.tar_model}.npy'

    np.save(os.path.join(args.loss_root, adv_save_name), adv_scores_dict)

    print(f'{args.tar_model} score saved')
