import os
import torch
from tqdm import tqdm
import wandb
from torchvision import utils

from faceswap.losses import compute_generator_loss, compute_discriminator_loss


class Trainer(object):

    def __init__(self, config, train_dataloader, test_dataloader, train_dataset_len,
                 gen_opt, discr_opt,
                 gen_scheduler, discr_scheduler,
                 stylegan_generator, discr,
                 landmark_encoder, target_encoder,
                 decoder, mapping_network):

        self.config = config
        self.train_dataset_len = train_dataset_len
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.stylegan_generator = stylegan_generator.to(self.device)
        self.discr = discr.to(self.device)

        self.gen_opt, self.discr_opt = gen_opt, discr_opt
        self.gen_scheduler, self.discr_scheduler = gen_scheduler, discr_scheduler

        self.landmark_encoder = landmark_encoder.to(self.device)
        self.target_encoder = target_encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.mapping_network = mapping_network.to(self.device)

    def train(self):
        self.target_encoder.train()
        self.decoder.train()
        self.landmark_encoder.train()

        for epoch in range(self.config.num_epochs):
            print(f'epoch = {epoch}')
            self.evaluate(epoch)
            self.train_epoch(epoch)



    def train_epoch(self, epoch):
        pbar = tqdm(self.train_dataloader)
        for iteration, (s_img, s_code, s_map, s_lmk, t_img, t_code, t_map, t_lmk, t_mask,
                        s_index, t_index) in enumerate(pbar):

            s_img = s_img.to(self.device)
            s_map = s_map.to(self.device).transpose(1, 3).float()
            t_img = t_img.to(self.device)
            t_map = t_map.to(self.device).transpose(1, 3).float()
            t_lmk = t_lmk.to(self.device)
            t_mask = t_mask.to(self.device)

            s_frame_code = s_code.to(self.device)
            t_frame_code = t_code.to(self.device)

            input_map = torch.cat([s_map, t_map], dim=1)
            t_mask = t_mask.unsqueeze_(1).float()

            t_lmk_code = self.landmark_encoder(input_map)

            zero_latent = torch.zeros((self.config.batch_size,
                                       18 - self.config.coarse,
                                       512)).to(self.device).detach()

            t_lmk_code = torch.cat([t_lmk_code, zero_latent], dim=1)
            fusion_code = s_frame_code + t_lmk_code

            fusion_code = torch.cat([fusion_code[:, : 18 - self.config.coarse],
                                     t_frame_code[:, 18 - self.config.coarse:]], dim=1)

            fusion_code = self.mapping_network(fusion_code.view(fusion_code.size(0), -1), 2)
            fusion_code = fusion_code.view(t_frame_code.size())

            source_feas = self.stylegan_generator([fusion_code],
                                                  input_is_latent=True, randomize_noise=False)
            target_feas = self.target_encoder(t_img)

            blend_img = self.decoder(source_feas, target_feas, t_mask)

            loss_g = compute_generator_loss(blend_img, s_img)
            loss_g.backward()

            self.gen_opt.step()

            # Discriminator training

            #self.discr_opt.zero_grad()

            #loss_d = compute_discriminator_loss(blend_img, t_img)
            #loss_d.backward()

            #self.discr_opt.step()

            if self.config.scheduler:
                self.gen_scheduler.step()
                #self.discr_scheduler.step()

            # Visualization

            total_loss = {
                'gen/loss_id': L_id.item(),
                'gen/loss_adv': L_adv.item(),
                'gen/loss_attr': L_attr.item(),
                'gen/loss_rec': L_rec.item(),

                'gen/loss_gen': lossG.item(),
                'discr/loss_discr': lossD.item(),
            }

            self.decoder.eval()

            step_to_log = epoch * self.train_dataset_len + (iteration + 1) * self.config.batch_size
            if (iteration + 1) % self.config.loss_log_step == 0:
                if self.config.use_wandb:
                    wandb.log(total_loss, step=step_to_log)
                else:
                    print(step_to_log, total_loss)

            if (iteration + 1) % self.config.model_save_step == 0:
                gen_path = os.path.join(self.config.model_save_dir,
                                        self.config.exp_name,
                                        f'{step_to_log}_generator.pth')
                gen_path_lattest = os.path.join(self.config.model_save_dir,
                                                self.config.exp_name,
                                                f'lattest_generator.pth')
                torch.save(self.decoder.state_dict(), gen_path)
                torch.save(self.discr.state_dict(), gen_path_lattest)

                d_path = os.path.join(self.config.model_save_dir,
                                      self.config.exp_name,
                                      f'{step_to_log}_discr.pth')
                d_path_lattest = os.path.join(self.config.model_save_dir,
                                              self.config.exp_name,
                                              f'lattest_generator.pth')
                torch.save(self.discr.state_dict(), d_path)
                torch.save(self.discr.state_dict(), d_path_lattest)
                
                
    def evaluate(self, epoch):
        for iteration, (s_img, s_code, s_map, s_lmk, t_img,
         t_code, t_map, t_lmk, t_mask, s_index, t_index) in enumerate(self.test_dataloader):
            if iteration > 5:
                break
                
            s_img = s_img.to(self.device)
            s_map = s_map.to(self.device).transpose(1, 3).float()
            t_img = t_img.to(self.device)
            t_map = t_map.to(self.device).transpose(1, 3).float()
            t_lmk = t_lmk.to(self.device)
            t_mask = t_mask.to(self.device)

            s_frame_code = s_code.to(self.device)
            t_frame_code = t_code.to(self.device)

            input_map = torch.cat([s_map, t_map], dim=1)
            t_mask = t_mask.unsqueeze_(1).float()

            t_lmk_code = self.landmark_encoder(input_map)

            zero_latent = torch.zeros((self.config.batch_size,
                                       18 - self.config.coarse,
                                       512)).to(self.device).detach()

            t_lmk_code = torch.cat([t_lmk_code, zero_latent], dim=1)
            fusion_code = s_frame_code + t_lmk_code

            fusion_code = torch.cat([fusion_code[:, : 18 - self.config.coarse],
                                     t_frame_code[:, 18 - self.config.coarse:]],
                                    dim=1)

            fusion_code = self.mapping_network(fusion_code.view(fusion_code.size(0), -1), 2)
            fusion_code = fusion_code.view(t_frame_code.size())

            source_feas = self.stylegan_generator([fusion_code],
                                                  input_is_latent=True, randomize_noise=False)
            target_feas = self.target_encoder(t_img)

            blend_img = self.decoder(source_feas, target_feas, t_mask)

            name = str(int(s_index[0]))+'_'+str(int(t_index[0]))
            with torch.no_grad():
                
                sample = torch.cat([s_img.detach(), t_img.detach()])
                sample = torch.cat([sample, blend_img.detach()])
                t_mask = torch.stack([t_mask,t_mask,t_mask],dim=1).squeeze(2)
                sample = torch.cat([sample, t_mask.detach()])

                utils.save_image(
                    sample,
                    f'./resulted_imgs/epoch_{epoch}_img_{name}.jpg',
                    nrow=4,
                    normalize=True,
                    range=(-1, 1),
                )