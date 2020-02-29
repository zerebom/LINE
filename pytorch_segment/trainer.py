import os
from pathlib import Path
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import MLFlowLogger, TensorBoardLogger
from PIL import Image
from torchvision import transforms

from unet import UNet3D
from dataset import DataPathMaker, KitsDataSet
from import_loss import BCEDiceLoss

import hydra
from omegaconf import DictConfig
import argparse
import cloudpickle


class KitsTrainer(pl.LightningModule):

    def __init__(self, hparams, tr_im, tr_lb, val_im, val_lb):
        super(KitsTrainer, self).__init__()
        # hparam経由では、listで渡すことは出来ない。
        self.hparams = hparams
        # TODO: yaml
        self.tr_DS = KitsDataSet(tr_im[:100], tr_lb[:100], phase='train', transform=None)
        self.val_DS = KitsDataSet(val_im[:100], val_lb[:100], phase='val', transform=None)
        self.devise = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = BCEDiceLoss(0.5, 0.5)
        self.batch_size = hparams.batch_size
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

        # 相対pathは/outputs/yyyy-mm-dd/hh-mm-ss/ の下からスタート
        weight_path = Path(f'./outputs/model_{self.current_epoch}.pkl')
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        Path(weight_path).touch()

        with open(weight_path, 'wb') as f:
            cloudpickle.dump(self.net.state_dict(), f)

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.parameters(), lr=0.001)
        return [torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.tr_DS, batch_size=self.hparams.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_DS, batch_size=self.hparams.batch_size)

#
@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig):
    p = argparse.ArgumentParser()

    # hparm はargsからしか保存できないので移し替える。
    args = p.parse_args()
    for key, value in cfg.model.items():
        args.__setattr__(key, value)
    print('cfg.save:', cfg.exp)
    tblogger = TensorBoardLogger(save_dir=cfg.exp.save_dir, name=cfg.exp.name)
    pacfg = cfg.patch
    trcfg = cfg.trainer

    train_path_df = DataPathMaker(pacfg.data_dir, patch_dir_name=pacfg.train_patch).\
        create_dataframe(pacfg.train_ids)
    val_path_df = DataPathMaker(pacfg.data_dir, patch_dir_name=pacfg.val_patch).\
        create_dataframe(pacfg.val_ids)

    tr_im_list = train_path_df[train_path_df['type'] == 'image']['path'].astype(str).values
    val_im_list = val_path_df[val_path_df['type'] == 'image']['path'].astype(str).values

    tr_lb_list = train_path_df[train_path_df['type'] == 'label']['path'].astype(str).values
    val_lb_list = val_path_df[val_path_df['type'] == 'label']['path'].astype(str).values

    model = KitsTrainer(args, tr_im_list, tr_lb_list, val_im_list, val_lb_list)

    # Called when the validation loop ends
    # checkpoint_callback = ModelCheckpoint(filepath ='ckpt', save_weights_only=True)

    trainer = pl.Trainer(
        # checkpoint_callback=checkpoint_callback,
        gpus=trcfg.gpus,
        row_log_interval=trcfg.row_log_interval,
        checkpoint_callback=False,
        logger=tblogger,
        max_epochs=trcfg.epoch,
        early_stop_callback=None)
    trainer.fit(model)


if __name__ == "__main__":

    main()
    # max_nb is deprecated
