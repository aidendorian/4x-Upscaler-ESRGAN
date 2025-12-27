from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import functional as F
import torch
from PIL import Image
import os
import random
from io import BytesIO

class ImageDataset(Dataset):
    def __init__(self, folder, hr_size=192, scale=4, training=True):
        self.hr_path = os.path.join('data', folder, 'hr_images')
        self.lr_path = os.path.join('data', folder, 'lr_images')

        self.hr_size = hr_size
        self.scale = scale
        self.lr_size = hr_size // scale
        self.training = training
        self.folder = folder

        self.filenames = sorted(os.listdir(self.hr_path))

    def __len__(self):
        return len(self.filenames)

    def apply_jpeg_compression(self, img, quality_range=(70, 95)):
        quality = random.randint(*quality_range)
        buffer = BytesIO()
        img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        hr = Image.open(os.path.join(self.hr_path, fname)).convert('RGB')
        lr = Image.open(os.path.join(self.lr_path, fname)).convert('RGB')        
        hr = hr.resize((self.hr_size, self.hr_size), Image.BICUBIC)
        lr = lr.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        if self.training:
            if self.folder in ['DIV2K', 'ISR']:
                jpeg_prob = 0.6
                noise_prob = 0.5
                jpeg_quality = (80, 95)
            else:
                jpeg_prob = 0.3
                noise_prob = 0.3
                jpeg_quality = (80, 95)
            
            if random.random() < jpeg_prob:
                lr = self.apply_jpeg_compression(lr, jpeg_quality)

            hr_t = F.to_tensor(hr)
            lr_t = F.to_tensor(lr)

            if random.random() < noise_prob:
                sigma = random.uniform(0.002, 0.007)
                lr_t = lr_t + torch.randn_like(lr_t) * sigma

            if random.random() < 0.5:
                hr_t = torch.flip(hr_t, [2])
                lr_t = torch.flip(lr_t, [2])
            
            if random.random() < 0.5:
                hr_t = torch.flip(hr_t, [1])
                lr_t = torch.flip(lr_t, [1])
            
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                hr_t = torch.rot90(hr_t, k, [1, 2])
                lr_t = torch.rot90(lr_t, k, [1, 2])
            
            hr_t = torch.clamp(hr_t, 0, 1)
            lr_t = torch.clamp(lr_t, 0, 1)
            
            return hr_t, lr_t
        
        else:
            random.seed(idx)
            
            if random.random() < 0.3:
                lr = self.apply_jpeg_compression(lr, quality_range=(88, 95))
            
            hr_t = F.to_tensor(hr)
            lr_t = F.to_tensor(lr)
            
            if random.random() < 0.2:
                sigma = 0.005
                lr_t = lr_t + torch.randn_like(lr_t) * sigma
            
            random.seed()
            hr_t = torch.clamp(hr_t, 0, 1)
            lr_t = torch.clamp(lr_t, 0, 1)
            return hr_t, lr_t


def get_dataloaders(batch_size=8,
                    num_workers=4,
                    pin_memory=True,
                    prefetch_factor=2,
                    persistent_workers=True):

    train_dataset = ConcatDataset([
        ImageDataset("DIV2K", hr_size=192, training=True),
        ImageDataset("Flickr2K", hr_size=192, training=True),
        ImageDataset("ISR", hr_size=192, training=True)
    ])

    val_dataset = ImageDataset("Validation", hr_size=192, training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader