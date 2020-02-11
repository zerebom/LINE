import pytorch_lightning as pl
import torch
from torch import nn
from .model import Generator,Discriminator
from .dataloder import 

class GAN(pl.LightningModule):
    def __init__(self, opt):
        super(GAN, self).__init__()

        # パラメータの保存
        self.scale_factor = opt.scale_factor
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size

        self.z_dim = 20
        self.mini_batch_size = 64

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ネットワーク定義
        self.net_G = Generator(z_dim=self.z_dim, image_size=64)
        self.net_D = Discriminator(z_dim=self.z_dim, image_size=64)

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        torch.backends.cudnn.benchmark = True

        # 誤差関数の定義
        self.criterion_MSE = nn.MSELoss()
        # self.criterion_VGG = VGGLoss(net_type='vgg19', layer='relu5_4')
        # self.criterion_GAN = GANLoss(gan_mode='wgangp')
        # self.criterion_TV = TVLoss()

    def forward(self, input):
        return self.net_G(input)

    def training_step(self, data_batch, batch_nb, optimizer_i):
        images = data_batch
        mini_batch_size = images.size()[0]

        if optimizer_i == 0:
            d_out_real = self.net_D(images)

            label_real = torch.full((mini_batch_size,), 1).to(self.device)
            label_fake = torch.full((mini_batch_size,), 0).to(self.device)

            input_z = torch.randn(self.mini_batch_size, self.z_dim).to(self.device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = self.net_G(input_z)
            d_out_fake = self.net_D(fake_images)

            d_loss_real = self.criterion(d_out_real.view(-1), label_real)
            d_loss_fake = self.criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            return {'loss': d_loss, 'prog': {'tng/d_loss': d_loss}}

        elif optimizer_i == 1:
            input_z = torch.randn(mini_batch_size, self.z_dim).to(self.device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = self.net_G(input_z)
            d_out_fake = self.net_D(fake_images)

            # 誤差を計算
            g_loss = self.criterion(d_out_fake.view(-1), label_real)

            return {'loss': g_loss, 'prog': {'tng/g_loss': g_loss}}

    
    @pl.data_loader
    def tng_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir='./data/DIV2K/train',
            scale_factor=self.scale_factor,
            patch_size=self.patch_size
        )
        return DataLoader(dataset, self.batch_size, shuffle=True, num_workers=4)




    def configure_optimzers(self):
        g_lr, d_lr = 0.0001, 0.0004
        beta1, beta2 = 0.0, 0.9
        optimizer_G = torch.optim.Adam(self.net_G.parameters(), g_lr, [beta1, beta2])
        optimizer_D = torch.optim.Adam(self.net_D.parameters(), d_lr, [beta1, beta2])

        return [optimizer_D, optimizer_G]
