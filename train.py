from math import log2

import config
import numpy as np
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import Discriminator, Generator
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import gradient_penalty, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmarks = True


def get_data(image_size):
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS)

    return dataloader


def wgan_gp_loss(real, fake, gp):
    return torch.mean(fake) - torch.mean(real) + config.LAMBDA_GP * gp + (0.001 * torch.mean(real ** 2))


def train(critic, generator, dataloader, step, alpha, optim_critic, optim_generator, scaler_generator, scaler_critic):
    loop = tqdm(dataloader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        current_batch_size = real.shape[0]

        # TRAIN CRITIC
        real = real.to(config.DEVICE)
        noise = torch.randn(current_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
        with torch.cuda.amp.autocast():
            fake = generator(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
            loss_critic = wgan_gp_loss(critic_real, critic_fake, gp)
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
        alpha += current_batch_size / ((config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataloader.dataset))
        alpha = min(alpha, 1)

        loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())

    return alpha


def save_prediction(generator, timestamp, img_size, epoch):
    prediction = generator(config.FIXED_NOISE)
    image = np.uint8((prediction.squeeze() + 1) * 255 / 2)
    x = Image.fromarray(image)
    x.save(f"{config.SAVE_PATH}/trainings/{timestamp}/generated/prediction_{img_size}_{epoch + 1}.jpg")


def main(timestamp, load_generator=None, load_critic=None):
    generator = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    critic = Discriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    optim_generator = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    optim_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    scaler_generator = torch.cuda.amp.GradScaler()
    scaler_critic = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(load_generator, generator, optim_generator, config.LEARNING_RATE)
        load_checkpoint(load_critic, critic, optim_critic, config.LEARNING_RATE)

    generator.train()
    critic.train()

    step = int(log2(config.START_TRAIN_AT_IMG_SIZE) / 4)

    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        img_size = 4 * 2 ** step
        dataloader = get_data(img_size)

        for epoch in range(num_epochs):
            print(f"========== {epoch + 1}/{num_epochs} ==========")
            alpha = train(critic, generator, dataloader, step, alpha, optim_critic, optim_generator, scaler_generator, scaler_critic)

            if config.SAVE_MODEL:
                filename_generator = f"{config.SAVE_PATH}/trainings/{timestamp}/generated/generator_{img_size}_{epoch + 1}.pth"
                filename_critic = f"{config.SAVE_PATH}/trainings/{timestamp}/generated/critic_{img_size}_{epoch + 1}.pth"
                save_checkpoint(generator, optim_generator, filename=filename_generator)
                save_checkpoint(critic, optim_critic, filename=filename_critic)

        step += 1


if __name__ == "__main__":
    main()
