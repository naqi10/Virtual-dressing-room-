import os
import cv2
import numpy as np
from PIL import Image
import torch


def gen_noise(size):
    if isinstance(size, torch.Size):
        size = tuple(size)
    noise = np.random.randn(*size).astype(np.float32)
    return torch.tensor(noise, dtype=torch.float32)


def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(array)
        im.save(os.path.join(save_dir, img_name), format='JPEG')


# ✅ Updated load_checkpoint: supports both dict and object checkpoints
def load_checkpoint(model, checkpoint_path, map_location='cpu'):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"'{checkpoint_path}' is not a valid checkpoint path")

    print(f"[info] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Handle possible checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Assume it's a plain state_dict
            state_dict = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    print("[info] Model weights loaded successfully.")

    # Return entire checkpoint if it includes optimizer/epoch etc.
    return checkpoint


# ✅ Optional: save checkpoint for training continuation
def save_checkpoint(model, optimizer=None, epoch=None, path='checkpoint.pth'):
    checkpoint = {'model': model.state_dict()}
    if optimizer:
        checkpoint['optimizer'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch

    torch.save(checkpoint, path)
    print(f"[info] Checkpoint saved to {path}")
