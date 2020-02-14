import torchvision
import pytorch_lightning as pl
import torch
from torch import nn
from model import Generator, Discriminator
from dataloder import make_datapath_list, ImageTransform, GAN_Img_Dataset
from torch.utils.data import DataLoader
import argparse
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

logger = TensorBoardLogger('/home/higuchi/ssd/Desktop/Sandbox/pytorch_GAN/lightning_logs', 'DGGAN')


class GAN(pl.LightningModule):
    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams
        # パラメータの保存
        # self.scale_factor = opt.scale_factor
        # self.batch_size = opt.batch_size
        # self.patch_size = opt.patch_size
        self.last_imgs = None

        self.z_dim = 20
        self.batch_size = 64

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ネットワーク定義
        self.net_G = Generator(z_dim=self.z_dim, image_size=64)
        self.net_D = Discriminator(z_dim=self.z_dim, image_size=64)

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        torch.backends.cudnn.benchmark = True

        # 誤差関数の定義
        self.criterion_MSE = nn.MSELoss()

    def forward(self, input):
        return self.net_G(input)

    def training_step(self, data_batch, batch_nb, optimizer_idx):
        images = data_batch
        self.last_imgs = images
        mini_batch_size = images.size()[0]
        label_real = torch.full((mini_batch_size,), 1).to(self.device)
        label_fake = torch.full((mini_batch_size,), 0).to(self.device)

        if optimizer_idx == 0:
            d_out_real = self.net_D(images)

            input_z = torch.randn(mini_batch_size, self.z_dim).to(self.device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = self.net_G(input_z)
            d_out_fake = self.net_D(fake_images)

            d_loss_real = self.criterion(d_out_real.view(-1), label_real)
            d_loss_fake = self.criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            tqdm_dict = {'d_loss': d_loss}

            return {'loss': d_loss, 'prog': {'tng/d_loss': d_loss}, 'log': tqdm_dict}

        elif optimizer_idx == 1:
            input_z = torch.randn(mini_batch_size, self.z_dim).to(self.device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = self.net_G(input_z)
            d_out_fake = self.net_D(fake_images)

            # 誤差を計算
            g_loss = self.criterion(d_out_fake.view(-1), label_real)

            return {'loss': g_loss, 'prog': {'tng/g_loss': g_loss}}

    def on_epoch_end(self):
        # save generated image
        z = torch.randn(self.batch_size, self.z_dim).to(self.device)
        z = z.view(z.size(0), z.size(1), 1, 1)
        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)
        sample_imgs = self.forward(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

    @pl.data_loader
    def train_dataloader(self):
        mean = (0.5,)
        std = (0.5,)
        dataset = GAN_Img_Dataset(
            file_list=make_datapath_list(), transform=ImageTransform(mean, std))
        return DataLoader(dataset, batch_size=self.batch_size)

    def configure_optimizers(self):
        # REQUIRED
        g_lr, d_lr = 0.0001, 0.0004
        beta1, beta2 = 0.0, 0.9
        optimizer_G = torch.optim.Adam(self.net_G.parameters(), g_lr, [beta1, beta2])
        optimizer_D = torch.optim.Adam(self.net_D.parameters(), d_lr, [beta1, beta2])

        return [optimizer_D, optimizer_G]

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=16, type=int)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=20, type=int)

        return parser


def main(hparams):
    # exp = Experiment(save_dir=f'./logs/')
    model = GAN(hparams)
    trainer = Trainer(max_epochs=200,
                      logger=logger,
                      gpus=hparams.gpus,
                      nb_gpu_nodes=hparams.nodes)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)

    parser = GAN.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams)
