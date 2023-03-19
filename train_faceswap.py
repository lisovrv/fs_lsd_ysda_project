import torch
import wandb

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import hydra
from omegaconf import DictConfig

from torchvision import transforms, utils
from dataset import *
from utils import *

from faceswap.models.encoders.psp_encoders import *
from faceswap.models.stylegan2.model import *
from faceswap.models.nets import *
from faceswap.trainer import Trainer


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    if config.use_wandb:
        run = wandb.init(config=config, project="fs_lsd_ysda", name="exp_0")
        run.log_code("./", include_fn=lambda path: path.endswith(".yaml"))

    to_tensor = transforms.Compose([
        transforms.Resize(config.size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = SourceICTargetICLM(config.image_path,
                                       to_tensor_256=to_tensor,
                                       to_tensor_1024=to_tensor)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  drop_last=True,
                                  num_workers=config.num_workers)

    test_dataloader = DataLoader(dataset=train_dataset,
                                 batch_size=4,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=4)

    encoder_lmk = GradualLandmarkEncoder(106 * 2)
    encoder_target = GPENEncoder(args.largest_size)

    decoder = Decoder(config.least_size, config.size)

    bald_model = F_mapping(mapping_lrmul=config.mapping_lrmul,
                           mapping_layers=config.mapping_layers,
                           mapping_fmaps=config.mapping_fmaps,
                           mapping_nonlinearity=config.mapping_nonlinearity)
    bald_model.eval()

    generator = Generator(config.size, config.latent, config.n_mlp)
    discr = Discriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d)

    if config.models.pretrained:
        e_ckpt = torch.load(config.models.e_ckpt, map_location=torch.device('cpu'))

        encoder_lmk.load_state_dict(e_ckpt["encoder_lmk"])
        encoder_target.load_state_dict(e_ckpt["encoder_target"])
        decoder.load_state_dict(e_ckpt["decoder"])
        generator.load_state_dict(e_ckpt["generator"])
        bald_model.load_state_dict(e_ckpt["bald_model"])

    gen_opt = torch.optim.Adam(generator.parameters(), config.optimizers.gen_lr,
                               betas=(0, 0.999), weight_decay=1e-4)
    discr_opt = torch.optim.Adam(discr.parameters(), config.optimizers.discr_lr,
                                 betas=(0, 0.999), weight_decay=1e-4)

    if config.scheduler:
        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_opt,
                                                        step_size=config.scheduler_step,
                                                        gamma=config.scheduler_gamma)
        discr_scheduler = torch.optim.lr_scheduler.StepLR(discr_opt,
                                                          step_size=config.scheduler_step,
                                                          gamma=config.scheduler_gamma)
    else:
        gen_scheduler, discr_scheduler = None, None

    # train
    trainer = Trainer(config, train_dataloader, test_dataloader,
                      generator, discr,
                      gen_opt, discr_opt,
                      gen_scheduler, discr_scheduler,
                      len(train_dataset),
                      encoder_lmk, encoder_target,
                      decoder, bald_model)
    trainer.train()


if __name__ == "__main__":
    main()
