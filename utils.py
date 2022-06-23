import torch
import numpy as np
from PIL import Image


def gradient_penalty(critic, real, fake, alpha, train_step):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).cuda()
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(inputs=interpolated_images, outputs=mixed_scores,
                                   grad_outputs=torch.ones_like(mixed_scores), create_graph=True, retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)

    return torch.mean((gradient_norm - 1) ** 2)


def save_prediction(fixed_noise, save_path, generator, alpha, step, img_size, epoch):
    prediction = generator(fixed_noise, alpha, step)
    prediction = prediction.squeeze().detach().cpu().numpy()
    prediction = prediction.transpose(1, 2, 0)
    img = np.uint8((prediction + 1) * 255 / 2)
    x = Image.fromarray(img).convert('RGB')
    x = x.resize((256, 256), resample=Image.NEAREST)
    x.save(f"{save_path}/trainings/{img_size}x{img_size}_{epoch}.jpg")


def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr