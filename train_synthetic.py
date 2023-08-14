import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from model.unet import UNet

from util.data_loading import LidarDatasetSynthetic


def train_model(
        model,
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

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                features, masks = batch['feature'], batch['mask']

                masks_pred = model(features)

                loss = criterion(masks_pred, masks.float())

                optimizer.zero_grad(set_to_none=True)

                loss.backward()
                optimizer.step()

                # number in this batch (not necessarily batch_size)
                pbar.update(features.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
    return model


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    model = UNet(3, 1, 64)

    # Training Parameters
    batch_size = 10
    learning_rate = 1e-5
    weight_decay = 1e-8
    momentum = 0.999
    pct_val = 0.1

    model = train_model(model, "data/synthetic", "data/synthetic",
                1, 5, 1e-5, 1e-8, 0.999, 0.1)
