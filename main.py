import torch
import argparse
import torchvision
import numpy as np
import tensorboardX
import os
from tqdm import tqdm
from models import PhotoAE
from dataloader import get_loader
from utils import *
import math

def get_train_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='test4', help="name")
    parser.add_argument("--type", type=str, default='train', help="[train, evaluate, visualize]")
    parser.add_argument("--dataset", type=str, default='CelebA', help="[CelebA, Synface]")
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint', help="location of checkpoint")
    parser.add_argument("--load_dir", type=str, help="location of load file")
    parser.add_argument("--tensorboard_dir", type=str, default='./tensorboard', help="location of tensorboard")
    parser.add_argument("--results_dir", type=str, default='./results', help="")
    parser.add_argument("--epoch", type=int, default=30, help="epoch")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--gpu_ids", type=str, default='4', help="GPU ids")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--workers", type=int, default=8, help="num workers")
    parser.add_argument("--train_type", type=int, default='0', help="iteration and batch size")
    parser.add_argument("--debug", action = 'store_true', help="for debugging")

    parser.add_argument("--ch_img", type=int, default=3, help="")
    parser.add_argument("--image_size", type=int, default=64, help="")
    parser.add_argument("--ch_latent", type=int, default=64, help="")
    parser.add_argument("--ch_mid_latent", type=int, default=128, help="")
    parser.add_argument("--save_point", type=int, default=100, help="")

    return parser.parse_args()

def train(opt, model, board) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_loader(opt, 'train')
    max_iter = len(dataloader)
    #dataloader_val = get_loader(opt,)
    gt_exist = False
    if opt.dataset == 'Synface':
        gt_exist = True

    for epoch in range(opt.epoch):
        print(f'training {epoch} epoch')
        model.to_train()
        for i, data in enumerate(tqdm(dataloader)):
            if gt_exist:
                img, gt = data
            else:
                img = data
            losses, results_images, results_scalars = model.train(img)
            if i % opt.save_point == 0:
                add_losses(board, losses, i + max_iter * epoch)
                add_images(board, results_images, i + max_iter * epoch)
                #add_scalars(board, results_scalars, i + max_iter * epoch)
                #add_images(board, val_images, i, 'Validation')
        if opt.dataset == 'Synface':
            dataloader_val = get_loader(opt, 'test')
            model.to_eval()
            num_data = dataloader_val.dataset.__len__()
            SIDE_total = 0
            MAD_total = 0
            for i, data in enumerate(tqdm(dataloader_val)):
                if gt_exist:
                    img, gt = data
                else:
                    img = data
                with torch.no_grad():
                    SIDE, MAD = model.evaluate(img, gt)
                SIDE_total += SIDE.sum()
                MAD_total += MAD.sum()
            SIDE_avg = SIDE_total / num_data
            MAD_avg = MAD_total / num_data
            print(f'SIDE_avg : {SIDE_avg}')
            print(f'MAD_avg : {MAD_avg}')
        if epoch % 1 == 0:
            model.save(opt, epoch)
    model.save(opt, epoch)

def evaluate(opt, model) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_loader(opt, 'test')
    max_iter = len(dataloader)
    gt_exist = False
    if opt.dataset == 'Synface':
        gt_exist = True

    num_data = dataloader.dataset.__len__()
    SIDE_total = 0
    MAD_total = 0
    for i, data in enumerate(tqdm(dataloader)):
        if gt_exist:
            img, gt = data
        else:
            img = data
        with torch.no_grad():
            SIDE, MAD = model.evaluate(img, gt)
        SIDE_total += SIDE.sum()
        MAD_total += MAD.sum()

    SIDE_avg = SIDE_total / num_data
    MAD_avg = MAD_total / num_data
    print(f'SIDE_avg : {SIDE_avg}')
    print(f'MAD_avg : {MAD_avg}')

def visualize(opt, model) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = os.path.join(opt.results_dir, os.path.join(opt.load_dir,'').split('/')[-3])
    dataloader = get_loader(opt, 'test')
    max_iter = len(dataloader)
    gt_exist = False
    if opt.dataset == 'Synface':
        gt_exist = True

    os.makedirs(os.path.join(results_dir, 'albedo'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'albedo_prime'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'depth_prime'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'conf'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'conf_prime'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'cannon_img'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'cannon_img_prime'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'img_hat'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'img_hat_prime'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'gt_img'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'recon_depth'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'recon_depth_masked'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'gt_masked'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'rotation_img'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'relight'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'img_with_sym_line'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'conf_prime_vis'), exist_ok=True)

    for i, data in enumerate(tqdm(dataloader)):
        if gt_exist:
            img, gt = data
        else:
            img = data
            gt = None
        with torch.no_grad():
            results_images = model.visualize(img, gt)
        batch_size = img.shape[0]

        ambience = torch.ones([batch_size, 1]).to(device) * 0.8
        diffuse = torch.ones([batch_size, 1]).to(device) * 0.3
        light_direction = torch.cuda.FloatTensor([0.9, 0.3, 0.93]).unsqueeze(0).repeat(batch_size, 1)
        canonical, shading = model.renderer.render_cannonical(results_images['albedo'], results_images['depth'], ambience, diffuse, light_direction)
        for j in range(batch_size):
            torchvision.utils.save_image((canonical[j] / 2 + 0.5).clamp(0,1), os.path.join(results_dir, 'relight', f'{i * img.shape[0] + j}.png'))

        for j, rotation_img in enumerate(results_images['recon_rotation_img']):
            torchvision.utils.save_image(rotation_img/2 + 0.5, os.path.join(results_dir, 'rotation_img',f'{i * img.shape[0] + j}.png'), nrow=6)

        for j in range(batch_size):
            if gt is not None:
                torchvision.utils.save_image(((results_images['gt'].unsqueeze(1)[j] - 0.9) / (1.1 - 0.9)).clamp(0, 1),
                                             os.path.join(results_dir, 'gt', f'{i * img.shape[0] + j}.jpg'))
                torchvision.utils.save_image(
                    ((results_images['gt_masked'].unsqueeze(1)[j] - 0.9) / (1.1 - 0.9)).clamp(0, 1),
                    os.path.join(results_dir, 'gt_masked', f'{i * img.shape[0] + j}.jpg'))
                torchvision.utils.save_image(
                    ((results_images['recon_depth'].unsqueeze(1)[j] - 0.9) / (1.1 - 0.9)).clamp(0, 1),
                    os.path.join(results_dir, 'recon_depth', f'{i * img.shape[0] + j}.jpg'))
                torchvision.utils.save_image(
                    ((results_images['recon_depth_masked'].unsqueeze(1)[j] - 0.9) / (1.1 - 0.9)).clamp(0, 1),
                    os.path.join(results_dir, 'recon_depth_masked', f'{i * img.shape[0] + j}.jpg'))

            torchvision.utils.save_image((results_images['depth'].unsqueeze(1)[j].clamp(0.9,1.1) - 0.9) / (1.1 -0.9), os.path.join(results_dir, 'depth', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['depth_prime'].unsqueeze(1)[j].clamp(0.9,1.1) - 0.9) / (1.1 -0.9), os.path.join(results_dir, 'depth_prime', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['img_hat'][j] / 2 + 0.5).clamp(0,1), os.path.join(results_dir, 'img_hat', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['img_hat_prime'][j] / 2 + 0.5).clamp(0,1), os.path.join(results_dir, 'img_hat_prime', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['albedo'][j] / 2 + 0.5).clamp(0,1), os.path.join(results_dir, 'albedo', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['albedo_prime'][j] / 2 + 0.5).clamp(0,1), os.path.join(results_dir, 'albedo_prime', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['conf'][j]).clamp(0,1), os.path.join(results_dir, 'conf', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['conf_prime'][j]).clamp(0,1), os.path.join(results_dir, 'conf_prime', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['cannon_img'][j] / 2 + 0.5).clamp(0,1), os.path.join(results_dir, 'cannon_img', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['cannon_img_prime'][j] / 2 + 0.5).clamp(0,1), os.path.join(results_dir, 'cannon_img_prime', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['gt_img'][j] / 2 + 0.5).clamp(0,1), os.path.join(results_dir, 'gt_img', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['img_with_sym_line'][j] / 2 + 0.5).clamp(0,1), os.path.join(results_dir, 'img_with_sym_line', f'{i * img.shape[0] + j}.jpg'))
            torchvision.utils.save_image((results_images['conf_prime_vis'][j] / 2 + 0.5).clamp(0,1), os.path.join(results_dir, 'conf_prime_vis', f'{i * img.shape[0] + j}.jpg'))


            #torchvision.utils.save_image(results_images['recon_depth'], os.path.join(results_dir, 'recon_depth', f'{i * img.shahpe[0] + j}.jpg'))
        if i ==3:
            break



if __name__=="__main__" :
    opt = get_train_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    model = PhotoAE(opt)
    if opt.load_dir is not None:
        model.load(opt)
    board = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    if opt.type == 'train':
        train(opt, model, board)
    elif opt.type == 'evaluate':
        model.to_eval()
        evaluate(opt, model)
    elif opt.type == 'visualize':
        model.to_eval()
        visualize(opt, model)
    else :
        raise NotImplementedError