from math import log2
import os

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import Critic, Generator
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import gradient_penalty, load_checkpoint, save_checkpoint, save_prediction
torch.backends.cudnn.benchmarks = True


def get_data(image_size, data_path, batch_sizes, num_workers):
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    batch_size = batch_sizes[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


def wgan_gp_loss(real, fake, lambda_gp, gp):
    return torch.mean(fake) - torch.mean(real) + lambda_gp * gp + (0.001 * torch.mean(real ** 2))


def train(critic, generator, z_dim, dataloader, step, alpha, lambda_gp, pro_epochs, optim_critic, optim_generator,
          scaler_generator, scaler_critic):

    loop = tqdm(dataloader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        current_batch_size = real.shape[0]

        # TRAIN CRITIC
        real = real.cuda()
        noise = torch.randn(current_batch_size, z_dim, 1, 1).cuda()
        with torch.cuda.amp.autocast():
            fake = generator(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step)
            loss_critic = wgan_gp_loss(critic_real, critic_fake, lambda_gp, gp)
        optim_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(optim_critic)
        scaler_critic.update()

        # TRAIN GENERATOR
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = - torch.mean(gen_fake)
        optim_generator.zero_grad()
        scaler_generator.scale(loss_gen).backward()
        scaler_generator.step(optim_generator)
        scaler_generator.update()

        # UPDATE ALPHA
        alpha += current_batch_size / ((pro_epochs[step] * 0.5) * len(dataloader.dataset))
        alpha = min(alpha, 1)

    return alpha


def main(start_img_size, num_updates, data_path, save_path, lr, batch_sizes, z_dim, in_channels, lambda_gp,
         pro_epochs, fixed_noise, num_workers, load_generator=None, load_critic=None):

    step = int(log2(start_img_size / 4))
    generator = Generator(z_dim, in_channels).cuda()
    critic = Critic(in_channels).cuda()
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.99))
    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.99))
    scaler_generator = torch.cuda.amp.GradScaler()
    scaler_critic = torch.cuda.amp.GradScaler()

    if load_generator is not None and load_critic is not None:
        load_checkpoint(load_generator, generator, optim_generator, lr)
        load_checkpoint(load_critic, critic, optim_critic, lr)

    generator.train()
    critic.train()

    print("Starting training loop...")

    for num_epochs in pro_epochs[step:step+num_updates+1]:
        alpha = 1e-5
        img_size = 4 * 2 ** step

        print(f"\nStarting image size: {img_size}x{img_size}")
        dataloader = get_data(img_size, data_path, batch_sizes, num_workers)

        # create the path to save the models if not exists
        if not os.path.exists(f"{save_path}/trainings/{img_size}x{img_size}"):
            os.mkdir(f"{save_path}/trainings/{img_size}x{img_size}")

        for epoch in range(1, num_epochs+1):
            print(f"\n========== {epoch}/{num_epochs} ==========")
            # save the current alpha to do the prediction
            current_alpha = alpha

            # train the model
            alpha = train(critic, generator, z_dim, dataloader, step, alpha, lambda_gp, pro_epochs,
                          optim_critic, optim_generator, scaler_generator, scaler_critic)

            # save the generator
            filename_generator = f"{save_path}/trainings/{img_size}x{img_size}/generator_{epoch}.pth"
            save_checkpoint(generator, optim_generator, filename=filename_generator)

            # save the critic only for the last
            if epoch == num_epochs:
                filename_critic = f"{save_path}/trainings/{img_size}x{img_size}/critic_{epoch}.pth"
                save_checkpoint(critic, optim_critic, filename=filename_critic)

            # save the prediction
            save_prediction(fixed_noise, save_path, generator, current_alpha, step, img_size, epoch)

            print(f"\nModel(s) + prediction saved!")

        step += 1
