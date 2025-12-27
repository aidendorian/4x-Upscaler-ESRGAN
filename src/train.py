import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from esrgan import RRDBNet, Discriminator
from losses import (pixel_loss, perception_loss, gan_loss_D, gan_loss_G, 
                    normalize_vgg, VGGFeatureExtractor, color_consistency_loss)
from dataloader import get_dataloaders
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from checkpoint import load_checkpoint, save_checkpoint
from torchvision.utils import save_image
import os
import random

HR_SIZE = 192
BATCH_SIZE = 10

PSNR_EPOCHS = 100
GAN_EPOCHS = 40
TOTAL_EPOCHS = PSNR_EPOCHS + GAN_EPOCHS 

LOSS_WEIGHTS = {
    'pixel': 1.0,
    'perceptual': 1.0,
    'gan': 0.005,
    'color': 0.05,
}

def get_gan_weight(epoch, start_epoch=PSNR_EPOCHS, warmup_epochs=20):
    if epoch < start_epoch:
        return 0.0
    
    gan_epoch = epoch - start_epoch
    if gan_epoch < warmup_epochs:
        return LOSS_WEIGHTS['gan'] * (gan_epoch / warmup_epochs)
    return LOSS_WEIGHTS['gan']

RESUME_FROM_CHECKPOINT = True
CHECKPOINT_PATH_G = 'checkpoints/generator_epoch_140.pth'
CHECKPOINT_PATH_D = 'checkpoints/discriminator_epoch_140.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('val_output', exist_ok=True)
os.makedirs('models', exist_ok=True)

train_data, val_data = get_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

ITERS_PER_EPOCH = len(train_data)

TARGET_PSNR_ITERS = ITERS_PER_EPOCH * PSNR_EPOCHS
TARGET_GAN_ITERS = ITERS_PER_EPOCH * GAN_EPOCHS
TOTAL_TARGET_ITERS = TARGET_PSNR_ITERS + TARGET_GAN_ITERS

generator = RRDBNet().to(device)
discriminator = Discriminator().to(device)
vgg = VGGFeatureExtractor().to(device).eval()

for param in vgg.parameters():
    param.requires_grad = False

scaler_G = GradScaler()
scaler_D = GradScaler()

g_opt = Adam(generator.parameters(), lr=2e-4, betas=(0.9, 0.999))
d_opt = Adam(discriminator.parameters(), lr=2e-4, betas=(0.9, 0.999))

decay_epochs_g = [PSNR_EPOCHS + int(GAN_EPOCHS * 0.5), 
                  PSNR_EPOCHS + int(GAN_EPOCHS * 0.75),
                  PSNR_EPOCHS + int(GAN_EPOCHS * 0.9)]

scheduler_G = MultiStepLR(g_opt, milestones=decay_epochs_g, gamma=0.5)
scheduler_D = MultiStepLR(d_opt, milestones=decay_epochs_g, gamma=0.5)

start_epoch = 0
global_iteration = 0
training_phase = 'psnr'
best_val_psnr = 0.0

if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH_G):
    print(f"Resuming Training")
    start_epoch, _, generator, g_opt, scaler_G, global_iteration = load_checkpoint(
        generator, g_opt, scaler_G, CHECKPOINT_PATH_G)
    
    if global_iteration is None:
        global_iteration = 0
    
    if start_epoch >= PSNR_EPOCHS:
        training_phase = 'gan'
        if os.path.exists(CHECKPOINT_PATH_D):
            _, _, discriminator, d_opt, scaler_D, _ = load_checkpoint(
                discriminator, d_opt, scaler_D, CHECKPOINT_PATH_D)
    
    for _ in range(start_epoch):
        scheduler_G.step()
        if training_phase == 'gan':
            scheduler_D.step()
    
    print(f'Resumed from Epoch {start_epoch}, Iteration {global_iteration:,}, Phase: {training_phase}')


def train_psnr_epoch(epoch, iteration):
    generator.train()
    epoch_loss = 0.0
    epoch_pix_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_data, desc=f"[PSNR] E{epoch+1}/{PSNR_EPOCHS} | Iter {iteration:,}/{TARGET_PSNR_ITERS:,}", leave=False)
    
    for hr_imgs, lr_imgs in pbar:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        
        generator.zero_grad(set_to_none=True)
        
        with autocast(device_type='cuda'):
            sr_imgs = generator(lr_imgs)            
            L_pix = pixel_loss(sr_imgs, hr_imgs)
            L_G = L_pix
        
        scaler_G.scale(L_G).backward()
        scaler_G.step(g_opt)
        scaler_G.update()
        
        epoch_loss += L_G.item()
        epoch_pix_loss += L_pix.item()
        num_batches += 1
        
        pbar.set_postfix({
            'Loss': f'{L_G.item():.4f}',
            'Pix': f'{L_pix.item():.4f}',
            'Iter': iteration
        })
        iteration += 1
    
    return epoch_loss / num_batches, epoch_pix_loss / num_batches, iteration


def train_gan_epoch(epoch, iteration):
    generator.train()
    discriminator.train()
    epoch_D_loss = 0.0
    epoch_G_loss = 0.0
    epoch_pix_loss = 0.0
    epoch_perc_loss = 0.0
    epoch_gan_loss = 0.0
    num_batches = 0
    
    current_gan_weight = get_gan_weight(epoch)
    
    gan_iter = iteration - TARGET_PSNR_ITERS
    pbar = tqdm(train_data, desc=f"[GAN] E{epoch+1}/{TOTAL_EPOCHS} | Iter {gan_iter:,}/{TARGET_GAN_ITERS:,}", leave=False)
    
    for hr_imgs, lr_imgs in pbar:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        discriminator.zero_grad(set_to_none=True)
        
        with autocast(device_type='cuda'):
            with torch.no_grad():
                sr_imgs = generator(lr_imgs)
            
            real_pred = discriminator(hr_imgs)
            fake_pred = discriminator(sr_imgs.detach())
            
            loss_D = gan_loss_D(real_pred, fake_pred)
        
        scaler_D.scale(loss_D).backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)  # Added gradient clipping
        scaler_D.step(d_opt)
        scaler_D.update()
        generator.zero_grad(set_to_none=True)
        
        with autocast(device_type='cuda'):
            sr_imgs = generator(lr_imgs)
            L_pix = pixel_loss(sr_imgs, hr_imgs)
            
            vgg_sr = vgg(normalize_vgg(sr_imgs))
            with torch.no_grad():
                vgg_hr = vgg(normalize_vgg(hr_imgs))
            L_percep = perception_loss(vgg_sr, vgg_hr)
            
            fake_pred = discriminator(sr_imgs)
            with torch.no_grad():
                real_pred = discriminator(hr_imgs)
            L_gan = gan_loss_G(real_pred, fake_pred)
            
            L_color = color_consistency_loss(sr_imgs, hr_imgs)
            
            L_G = (LOSS_WEIGHTS['pixel'] * L_pix + 
                   LOSS_WEIGHTS['perceptual'] * L_percep + 
                   current_gan_weight * L_gan +
                   LOSS_WEIGHTS['color'] * L_color)
        
        scaler_G.scale(L_G).backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
        scaler_G.step(g_opt)
        scaler_G.update()
        
        epoch_D_loss += loss_D.item()
        epoch_G_loss += L_G.item()
        epoch_pix_loss += L_pix.item()
        epoch_perc_loss += L_percep.item()
        epoch_gan_loss += L_gan.item()
        num_batches += 1
        
        pbar.set_postfix({
            'D': f'{loss_D.item():.4f}',
            'G': f'{L_G.item():.4f}',
            'GAN_w': f'{current_gan_weight:.4f}',
            'Iter': iteration
        })
        
        iteration += 1
    
    return (epoch_D_loss / num_batches, epoch_G_loss / num_batches, 
            epoch_pix_loss / num_batches, epoch_perc_loss / num_batches, 
            epoch_gan_loss / num_batches, iteration)


def validate(epoch):
    generator.eval()
    val_losses = []
    
    with torch.no_grad():
        val_pbar = tqdm(val_data, desc="Validation", leave=False)
        for hr_val, lr_val in val_pbar:
            lr_val = lr_val.to(device, non_blocking=True)
            hr_val = hr_val.to(device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                sr_val = generator(lr_val)
                val_loss = pixel_loss(sr_val, hr_val)
                val_losses.append(val_loss.item())
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_psnr = 10 * torch.log10(torch.tensor(1.0 / (avg_val_loss ** 2)))
    
    print(f"  Validation PSNR: {val_psnr:.2f} dB | L1: {avg_val_loss:.4f}")
    
    index = random.randint(0, len(hr_val) - 1)
    save_image(sr_val[index].clamp(0, 1), f'val_output/epoch_{epoch+1}_sr.png')
    save_image(hr_val[index].clamp(0, 1), f'val_output/epoch_{epoch+1}_hr.png')
    save_image(lr_val[index].clamp(0, 1), f'val_output/epoch_{epoch+1}_lr.png')
    
    torch.save(generator.state_dict(), 'models/ESRGAN_latest.pth')
    torch.save(discriminator.state_dict(), 'models/ESRGAN_Discriminator_latest.pth')
    
    return val_psnr.item()

for epoch in range(start_epoch, TOTAL_EPOCHS):
    if epoch < PSNR_EPOCHS:
        # Phase 1: PSNR training
        avg_loss, avg_pix, global_iteration = train_psnr_epoch(epoch, global_iteration)
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} [PSNR] [Iter {global_iteration:,}/{TARGET_PSNR_ITERS:,}] - "
              f"Loss: {avg_loss:.4f} | Pixel: {avg_pix:.4f} "
              f"LR: {g_opt.param_groups[0]['lr']:.2e}")
        
        save_checkpoint(
            generator, g_opt, scaler_G, epoch + 1,
            'checkpoints', f'psnr_generator_epoch_{epoch+1}.pth',
            avg_loss, global_iteration
        )
        
        if (epoch + 1) == PSNR_EPOCHS:
            torch.save(generator.state_dict(), 'models/ESRGAN_PSNR.pth')
            print(f"\n>>> PSNR phase complete! Model saved to 'models/ESRGAN_PSNR.pth'\n")
    
    else:
        # Phase 2: GAN training
        avg_D, avg_G, avg_pix, avg_perc, avg_gan, global_iteration = train_gan_epoch(epoch, global_iteration)
        gan_iter = global_iteration - TARGET_PSNR_ITERS
        current_gan_weight = get_gan_weight(epoch)
        
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} [GAN] [Iter {gan_iter:,}/{TARGET_GAN_ITERS:,}] - "
              f"D: {avg_D:.4f} | G: {avg_G:.4f} | "
              f"Pixel: {avg_pix:.4f} | Percep: {avg_perc:.4f} | GAN: {avg_gan:.4f} (w={current_gan_weight:.4f}) | "
              f"LRg: {g_opt.param_groups[0]['lr']:.2e} | LRd: {d_opt.param_groups[0]['lr']:.2e}")
        
        save_checkpoint(
            generator, g_opt, scaler_G, epoch + 1,
            'checkpoints', f'generator_epoch_{epoch+1}.pth',
            avg_G, global_iteration
        )

        save_checkpoint(
            discriminator, d_opt, scaler_D, epoch + 1,
            'checkpoints', f'discriminator_epoch_{epoch+1}.pth',
            avg_D, global_iteration
        )
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        val_psnr = validate(epoch)
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(generator.state_dict(), 'models/ESRGAN_best.pth')
            print(f"  ★ New best model saved! PSNR: {val_psnr:.2f} dB")
    
    scheduler_G.step()
    if epoch >= PSNR_EPOCHS:
        scheduler_D.step()

torch.save(generator.state_dict(), 'models/ESRGAN_final.pth')
torch.save(discriminator.state_dict(), 'models/ESRGAN_Discriminator_final.pth')
print("Training Complete!")
print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")