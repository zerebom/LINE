import pytorch_lightning as pl


class GAN(pl.LightningModule):
   def __init__(self, opt):

        super(SRGANModel, self).__init__()

        # パラメータの保存
        self.scale_factor = opt.scale_factor
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size

        # ネットワーク定義
        self.net_G = Generator(z_dim=z_dim, image_size=64)
        self.net_D = Discriminator(z_dim=z_dim, image_size=64)

        # 誤差関数の定義
        # self.criterion_MSE = nn.MSELoss()
        # self.criterion_VGG = VGGLoss(net_type='vgg19', layer='relu5_4')
        # self.criterion_GAN = GANLoss(gan_mode='wgangp')
        # self.criterion_TV = TVLoss()
