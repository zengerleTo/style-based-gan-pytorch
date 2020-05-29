import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator, PortraitEncoder


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train(args, dataset, encoder, generator, discriminator):
    step=5
    resolution = 128
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    adjust_lr(e_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(560_000))

    requires_grad(encoder, True)
    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    enc_loss_val = 0
    grad_loss_val = 0

    alpha = 1
    used_sample = 0
    epoch = 0

    for i in pbar:
        if used_sample > args.phase:
            used_sample = 0
            epoch += 1

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)

            torch.save(
                {
                    'encoder': encoder.module.state_dict(),
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'e_optimizer': e_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                },
                f'checkpoint/epoch-{epoch}.model',
            )

            adjust_lr(e_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))
            
        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()

        #discriminator training
#        discriminator.zero_grad()
#
#        if args.loss == 'wgan-gp':
#            real_predict = discriminator(real_image, step=step, alpha=alpha)
#            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
#            (-real_predict).backward()
#
#        elif args.loss == 'r1':
#            real_image.requires_grad = True
#            real_scores = discriminator(real_image, step=step, alpha=alpha)
#            real_predict = F.softplus(-real_scores).mean()
#            real_predict.backward(retain_graph=True)
#
#            grad_real = grad(
#                outputs=real_scores.sum(), inputs=real_image, create_graph=True
#            )[0]
#            grad_penalty = (
#                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
#            ).mean()
#            grad_penalty = 10 / 2 * grad_penalty
#            grad_penalty.backward()
#            if i%10 == 0:
#                grad_loss_val = grad_penalty.item()
#
#        if args.mixing and random.random() < 0.9:
#            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
#                4, b_size, code_size, device='cuda'
#            ).chunk(4, 0)
#            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
#            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
#
#        else:
#            gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(
#                2, 0
#            )
#            gen_in1 = gen_in1.squeeze(0)
#            gen_in2 = gen_in2.squeeze(0)
#
#        fake_image = generator(gen_in1, step=step, alpha=alpha)
#        fake_predict = discriminator(fake_image, step=step, alpha=alpha)
#
#        if args.loss == 'wgan-gp':
#            fake_predict = fake_predict.mean()
#            fake_predict.backward()
#
#            eps = torch.rand(b_size, 1, 1, 1).cuda()
#            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
#            x_hat.requires_grad = True
#            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
#            grad_x_hat = grad(
#                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
#            )[0]
#            grad_penalty = (
#                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
#            ).mean()
#            grad_penalty = 10 * grad_penalty
#            grad_penalty.backward()
#            if i%10 == 0:
#                grad_loss_val = grad_penalty.item()
#                disc_loss_val = (-real_predict + fake_predict).item()
#
#        elif args.loss == 'r1':
#            fake_predict = F.softplus(fake_predict).mean()
#            fake_predict.backward()
#            if i%10 == 0:
#                disc_loss_val = (real_predict + fake_predict).item()
#
#        d_optimizer.step()

        #Encoder training
        encoder.zero_grad()

        requires_grad(encoder, True)
        requires_grad(discriminator, False)

        latent_w = encoder(real_image)
        
        embedded_image = generator(latent_w, step=step, alpha=alpha)

        predict = discriminator(embedded_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            loss = -predict.mean()

        elif args.loss == 'r1':
            loss = F.softplus(-predict).mean()
            
        loss += nn.MSELoss(real_image, embedded_image)

        if i%10 == 0:
            enc_loss_val = loss.item()

        loss.backward()
        e_optimizer.step()

        requires_grad(encoder, False)
        requires_grad(discriminator, True)

        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))

            with torch.no_grad():
                for _ in range(gen_i):
                    images.append(
                        generator(
                            encoder(real_image), step=step, alpha=alpha
                        ).data.cpu()
                    )

            utils.save_image(
                torch.cat(images, 0),
                f'sample/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {enc_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=56_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )
    parser.add_argument('--gen_path', default='', type=str, help='path of the pretrained generator')
    parser.add_argument('--discr_path', default='',type=str, help='path of the pretrained generator')

    args = parser.parse_args()

    encoder = nn.DataParallel(PortraitEncoder(size=128, filters=64, filters_max=512, num_layers=1)).cuda()
    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()

    e_optimizer = optim.Adam(
        encoder.module.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
#    e_optimizer.add_param_group(
#        {
#            'params': encoder.module.parameters(),
#            'lr': args.lr * 0.01,
#            'mult': 0.01,
#        }
#    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    if args.gen_path != '':
        generator.module.load_state_dict(torch.load(args.gen_path))
    if args.discr_path != '':
        discriminator.module.load_state_dict(torch.load(args.discr_path)['discriminator'])

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)

        encoder.module.load_state_dict(ckpt['encoder'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        e_optimizer.load_state_dict(ckpt['e_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    train(args, dataset, encoder, generator, discriminator)
