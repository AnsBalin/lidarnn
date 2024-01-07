import logging
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from model.unet import UNet

from util.data_loading import LidarDatasetSynthetic


def dice_coeff(input, target, reduce_batch_first=False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def train_model(
        model,
        device,
        features_path,
        masks_path,
        epochs: int = 10,
        batch_size: int = 5,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        pct_val: float = 0.1,

):
    dataset = LidarDatasetSynthetic(features_path, masks_path, 1000)
    n_val = int(len(dataset) * pct_val)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(0))
    loader_args = dict(batch_size=batch_size)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False,
                            drop_last=True, **loader_args)

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay,
                              momentum=momentum,
                              foreach=True)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        i = 1
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)

                features, masks = batch['feature'], batch['mask']

                features = features.to(
                    device='cuda', dtype=torch.float32, memory_format=torch.channels_last)
                masks = masks.to(device='cuda', dtype=torch.float32)

                with torch.autocast(device_type='cuda', enabled=True):
                    masks_pred = model(features)
                    loss = criterion(masks_pred.squeeze(1), masks.float())
                    loss += 1. - \
                        dice_coeff(F.sigmoid(masks_pred.squeeze(1)),
                                   masks.float())

                if math.isnan(loss.item()):
                    return model

                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                epoch_loss += loss.item()
                # number in this batch (not necessarily batch_size)
                pbar.update(features.shape[0])
                pbar.set_postfix(**{'loss (batch)': epoch_loss / i})
                i = i + 1

    return model


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(3, 1, 64)
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    # Training Parameters
    batch_size = 10
    learning_rate = 1e-5
    weight_decay = 1e-8
    momentum = 0.999
    pct_val = 0.1

    model = train_model(model, device, "data/synthetic", "data/synthetic",
                        1, 5, 1e-5, 1e-8, 0.999, 0.1)
