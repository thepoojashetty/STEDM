# on windows, openslide needs to be installed
# but even then, we have to manually tell the system where to find it
import os

import hydra
import torch
import pytorch_lightning as pl

from pathlib import Path
from datetime import timedelta
from omegaconf import DictConfig
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy

from modules.ldm_diffusion import LDM_Diffusion
from data.dm import DataModule

import numpy as np
from PIL import Image
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers


@hydra.main(version_base=None, config_path="conf", config_name="config_predict")
def main(cfg : DictConfig):
    print("Predicting Diffusion")
    # create ckpt path
    if hasattr(cfg, "ckpt_name"):
        ckpt_name = cfg.ckpt_name
    else:
        return
    ckpt_path = cfg.location.result_dir + "/checkpoints/" + ckpt_name

    #initialize logger
    run_name = cfg.run_name if hasattr(cfg, "run_name") else f"Predict_Diff_{cfg.data.name}_{cfg.style_sampling.name}_epoch50_cfg{cfg.cfg_scale}"
    logger = pl_loggers.WandbLogger(project="Semantic Style Diffusion", name=run_name)

    # delete pretrained ckpt path
    # del cfg.diffusion.ckpt_path
    # load module
    module = LDM_Diffusion.load_from_checkpoint(ckpt_path, cfg=cfg, map_location=torch.device("cpu"), strict=False)
    # module = LDM_Diffusion.load_from_checkpoint(ckpt_path)

    # set float point precision
    torch.set_float32_matmul_precision('high')

    #dataset cfg
    for name in cfg.data.dataset:
        ds_name = name
    print("ds_name", ds_name,"\n")
    ds_config = cfg.data.dataset[ds_name]

    # load an image and predict
    test_folder_path = cfg.location.data_dir + "/" + cfg.data.test_folder
    # load test condition image
    test_img = np.array(Image.open(test_folder_path + "/test_c2.png").convert('L'))
    test_img = (test_img == 26).astype(np.uint8)

    out_ch = 2
    c = F.one_hot(torch.from_numpy(test_img).to(module.device).to(torch.long), num_classes=out_ch).unsqueeze(0).to(torch.float32)

    c= torch.zeros_like(c)

    # load style images
    test_style_path = test_folder_path + "/" + cfg.style_sampling.name

    with torch.no_grad():
        # set to validation
        module._model.eval()

        if cfg.style_sampling.name == "augmented":
            #x/127.5-1 gets it to the range (-1,1)
            style_0 = (torch.from_numpy(np.array(Image.open(test_style_path + "/0_img.png"))[:,:,:3]).to(torch.float32).to(module.device).unsqueeze(0).unsqueeze(0) / 127.5)-1
            style_1 = (torch.from_numpy(np.array(Image.open(test_style_path + "/1_img.png"))[:,:,:3]).to(torch.float32).to(module.device).unsqueeze(0).unsqueeze(0) / 127.5)-1
            style_2 = (torch.from_numpy(np.array(Image.open(test_style_path + "/2_img.png"))[:,:,:3]).to(torch.float32).to(module.device).unsqueeze(0).unsqueeze(0) / 127.5)-1
            style_3 = (torch.from_numpy(np.array(Image.open(test_style_path + "/3_img.png"))[:,:,:3]).to(torch.float32).to(module.device).unsqueeze(0).unsqueeze(0) / 127.5)-1
        else:
            style_0 = (torch.zeros((1,1,512,512,3), dtype=torch.float32, device=module.device) / 127.5)-1
            style_1 = style_0
            style_2 = style_0
            style_3 = style_0

        ldm_batch_0 = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=module.device), "segmentation": c, "style_imgs":style_0}
        ldm_batch_1 = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=module.device), "segmentation": c, "style_imgs":style_1}
        ldm_batch_2 = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=module.device), "segmentation": c, "style_imgs":style_2}
        ldm_batch_3 = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=module.device), "segmentation": c, "style_imgs":style_3}
        
        # create conditioning tensor
        z, c_0 = module._model.get_input(ldm_batch_0, "image")
        z, c_1 = module._model.get_input(ldm_batch_1, "image")
        z, c_2 = module._model.get_input(ldm_batch_2, "image")
        z, c_3 = module._model.get_input(ldm_batch_3, "image")

        # get uncond conditioning
        ldm_batch_uncond = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=module.device), "segmentation": c, "style_imgs":torch.zeros_like(style_0)-2}
        z, c_uncond = module._model.get_input(ldm_batch_uncond, "image")

        for w in cfg.cfg_scale:
            # create predict output directory
            print("w", w,"\n")
            predict_dir = cfg.location.data_dir + "/syn_data_diff_w/" + f"{cfg.data.name}_{cfg.style_sampling.name}_epoch50_cfg{w}"
            Path(predict_dir).mkdir(parents=True, exist_ok=True)

            #sample images
            img_ddim_w0, _ = module._model.sample_log(c_0, batch_size=1, ddim=True, ddim_steps=cfg.ddim_steps, eta=cfg.eta, log_every_t=1000, unconditional_conditioning=c_uncond, unconditional_guidance_scale=w)
            img_ddim_w1, _ = module._model.sample_log(c_1, batch_size=1, ddim=True, ddim_steps=cfg.ddim_steps, eta=cfg.eta, log_every_t=1000, unconditional_conditioning=c_uncond, unconditional_guidance_scale=w)
            img_ddim_w2, _ = module._model.sample_log(c_2, batch_size=1, ddim=True, ddim_steps=cfg.ddim_steps, eta=cfg.eta, log_every_t=1000, unconditional_conditioning=c_uncond, unconditional_guidance_scale=w)
            img_ddim_w3, _ = module._model.sample_log(c_3, batch_size=1, ddim=True, ddim_steps=cfg.ddim_steps, eta=cfg.eta, log_every_t=1000, unconditional_conditioning=c_uncond, unconditional_guidance_scale=w)
                
            img_ddim_w0 = torch.clip(module._model.decode_first_stage(img_ddim_w0), -1, 1)
            img_ddim_w1 = torch.clip(module._model.decode_first_stage(img_ddim_w1), -1, 1)
            img_ddim_w2 = torch.clip(module._model.decode_first_stage(img_ddim_w2), -1, 1)
            img_ddim_w3 = torch.clip(module._model.decode_first_stage(img_ddim_w3), -1, 1)

            img_ddim_w0 = ((img_ddim_w0[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            img_ddim_w1 = ((img_ddim_w1[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            img_ddim_w2 = ((img_ddim_w2[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            img_ddim_w3 = ((img_ddim_w3[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)

            # log images to wandb
            logger.log_image(f"Sample Images cfg_scale{w}", images=[img_ddim_w0, img_ddim_w1, img_ddim_w2, img_ddim_w3], caption=["Test 0", "Test 1", "Test 2", "Test 3"])

            # save img
            Image.fromarray(img_ddim_w0).save(predict_dir + f"/img_00.png")
            Image.fromarray(img_ddim_w1).save(predict_dir + f"/img_01.png")
            Image.fromarray(img_ddim_w2).save(predict_dir + f"/img_02.png")
            Image.fromarray(img_ddim_w3).save(predict_dir + f"/img_03.png")

            # save seg
            seg_0 = torch.argmax(ldm_batch_0["segmentation"], dim=-1).cpu().numpy().astype(np.uint8).squeeze(0)
            seg_1 = torch.argmax(ldm_batch_1["segmentation"], dim=-1).cpu().numpy().astype(np.uint8).squeeze(0)
            seg_2 = torch.argmax(ldm_batch_2["segmentation"], dim=-1).cpu().numpy().astype(np.uint8).squeeze(0)
            seg_3 = torch.argmax(ldm_batch_3["segmentation"], dim=-1).cpu().numpy().astype(np.uint8).squeeze(0)
            Image.fromarray(seg_0).save(predict_dir + f"/seg_00.png")
            Image.fromarray(seg_1).save(predict_dir + f"/seg_01.png")
            Image.fromarray(seg_2).save(predict_dir + f"/seg_02.png")
            Image.fromarray(seg_3).save(predict_dir + f"/seg_03.png")


if __name__ == "__main__":
    print("main:Predicting Diffusion")
    main()