import torch
import os

def save_checkpoint(model, optimizer, scaler, epoch, save_dir, filename, loss=None, iteration=None):
    """Save complete training checkpoint with GradScaler"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'iteration': iteration,
        'rng_state': torch.get_rng_state(),
    }
    
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, scaler, checkpoint_path):
    """Load complete training checkpoint with GradScaler"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, None, model, optimizer, scaler
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("GradScaler state loaded")
    else:
        print("Warning: No GradScaler state found in checkpoint")

    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    iteration = checkpoint.get('iteration', None)
    
    return epoch, loss, model, optimizer, scaler, iteration