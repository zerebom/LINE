import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import pytorch_lightning as pl

from pytorch_lightning.logging import MLFlowLogger, TensorBoardLogger
from PIL import Image
from torchvision import transforms

from unet import UNet3D
from dataset import DataPathMaker, KitsDataSet
from import_loss import BCEDiceLoss

import hydra
from omegaconf import DictConfig

logger = TensorBoardLogger(save_dir='./tb_logger', name='tutorial')

DATA_DIR = '/home/higuchi/ssd/kits19/data'
train_patch = 'tumor_48x48x16'
val_patch = 'tumor_60x60x20'
train_ids = ['001']
val_ids = ['002']

train_path_df = DataPathMaker(DATA_DIR, patch_dir_name=train_patch).create_dataframe(train_ids)
val_path_df = DataPathMaker(DATA_DIR, patch_dir_name=val_patch).create_dataframe(val_ids)

train_im_list = train_path_df[train_path_df['type'] == 'image']['path'].astype(str).values
val_im_list = train_path_df[train_path_df['type'] == 'image']['path'].astype(str).values

train_lb_list = train_path_df[train_path_df['type'] == 'label']['path'].astype(str).values
val_lb_list = train_path_df[train_path_df['type'] == 'label']['path'].astype(str).values


class KitsTrainer(pl.LightningModule):

    def __init__(self, hparams):
        super(KitsTrainer, self).__init__()
        self.hparams = hparams
        # TODO: yaml
        self.tr_DS = KitsDataSet(train_im_list[:100], train_lb_list[:100], phase='train', transform=None)
        self.val_DS = KitsDataSet(val_im_list[:100], val_lb_list[:100], phase='val', transform=None)
        self.devise = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = BCEDiceLoss(0.5, 0.5)
        self.batch_size = 16

        self.net = UNet3D([self.batch_size, 1, 16, 48, 48], 3).cuda()

    def forward(self, im):
        return self.net(im)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        img, mask = img.cuda().float(), mask.cuda().float()
        out = self.forward(img)
        # print('out,mask:', out.size(), mask.size())

        loss_val = self.criterion(out, mask)

        if self.global_step % self.trainer.row_log_interval == 0:
            grid = torchvision.utils.make_grid(out[:, :, 1, :, :])
            self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

        tensorboard_logs = {'train_loss': loss_val}
        return {'loss': loss_val, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        if self.on_gpu:
            img = img.cuda(img.device.index)
        # img, mask = img.float(), mask.float()
        img, mask = img.cuda().float(), mask.cuda().float()
        out = self.forward(img)
        loss_val = self.criterion(out, mask)

        return {'val_loss': loss_val}

    # outputsはvalidation_stepのreturnのこと.モデルの出力ではない
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.parameters(), lr=0.001)
        return [torch.optim.Adam(self.net.parameters(), lr=0.001)]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.tr_DS, batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_DS, batch_size=self.batch_size)

@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig):
    model = KitsTrainer(hparams)


if __name__ == "__main__":

    model = KitsTrainer()
    # max_nb is deprecated
    trainer = pl.Trainer(
        gpus=2,
        row_log_interval=10,
        logger=logger,
        max_epochs=5,
        early_stop_callback=None)
    trainer.fit(model)
