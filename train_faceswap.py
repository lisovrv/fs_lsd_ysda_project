import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import wandb
import hydra
from omegaconf import DictConfig

from torch.utils.data import DataLoader
from torchvision import transforms

from faceswap.models.encoders.psp_encoders import GradualLandmarkEncoder
from faceswap.models.stylegan2.model import GPENEncoder, Decoder, Generator, Discriminator
# from faceswap.models.discriminator import Discriminator
from faceswap.models.nets import F_mapping

from faceswap.dataset import CelebaHqDataset
from faceswap.trainer import Trainer


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    if config.use_wandb:
        run = wandb.init(config=config, project="fs_lsd_ysda", name="exp_8_hm")
        run.log_code("./", include_fn=lambda path: path.endswith(".yaml"))

    to_tensor = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = CelebaHqDataset(config.dataset.image_path,
                              to_tensor_256=to_tensor,
                              to_tensor_1024=to_tensor,
                              dataset_size=1e5)

    train_dataloader = DataLoader(dataset=dataset_train,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=config.num_workers)
    
    dataset_test = CelebaHqDataset(config.dataset.image_path,
                              to_tensor_256=to_tensor,
                              to_tensor_1024=to_tensor,
                              shuffle=False,
                              dataset_size=1e2)

    test_dataloader = DataLoader(dataset=dataset_test,
                                 batch_size=2,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=0)

    landmark_encoder = GradualLandmarkEncoder(106 * 2)
    target_encoder = GPENEncoder(config.largest_size)
    decoder = Decoder(config.least_size, config.image_size)

    mapping_network = F_mapping(mapping_lrmul=config.mapping_lrmul,
                                mapping_layers=config.mapping_layers,
                                mapping_fmaps=config.mapping_fmaps,
                                mapping_nonlinearity=config.mapping_nonlinearity).eval()
    stylegan_generator = Generator(config.image_size, config.latent, config.n_mlp).eval()

    #discr = Discriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d)
    discr = Discriminator(size=1024)

    e_ckpt = torch.load(config.models.e_ckpt, map_location=torch.device('cpu'))
    if config.models.pretrained:
        landmark_encoder.load_state_dict({key[7:]: value for key, value in e_ckpt["encoder_lmk"].items()})
        target_encoder.load_state_dict({key[7:]: value for key, value in e_ckpt["encoder_target"].items()})
        decoder.load_state_dict({key[7:]: value for key, value in e_ckpt["decoder"].items()})
        
    stylegan_generator.load_state_dict({key[7:]: value for key, value in e_ckpt["generator"].items()})
    mapping_network.load_state_dict({key[7:]: value for key, value in e_ckpt["bald_model"].items()})
    
    gen_opt = torch.optim.Adam([{'params': landmark_encoder.parameters()},
                                {'params': target_encoder.parameters()},
                                {'params': decoder.parameters()},],
                                config.optimizers.gen_lr)
    
    
    discr_opt = torch.optim.Adam(discr.parameters(), config.optimizers.discr_lr)

    if config.scheduler:
        gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_opt,
                                                        step_size=config.scheduler_step,
                                                        gamma=config.scheduler_gamma)
        discr_scheduler = torch.optim.lr_scheduler.StepLR(discr_opt,
                                                          step_size=config.scheduler_step,
                                                          gamma=config.scheduler_gamma)
    else:
        gen_scheduler, discr_scheduler = None, None

    trainer = Trainer(config,
                      train_dataloader, test_dataloader, len(dataset_train),
                      gen_opt, discr_opt,
                      gen_scheduler, discr_scheduler,
                      stylegan_generator, discr,
                      landmark_encoder, target_encoder,
                      decoder, mapping_network)
    trainer.train()


if __name__ == "__main__":
    main()
