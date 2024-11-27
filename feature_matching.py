# on windows, openslide needs to be installed
# but even then, we have to manually tell the system where to find it
import os

import hydra
import torch
import pytorch_lightning as pl

from pathlib import Path
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
    print("Feature Matching")
    # create ckpt path
    if hasattr(cfg, "ckpt_name"):
        ckpt_name = cfg.ckpt_name
    else:
        return
    ckpt_path = cfg.location.result_dir + "/checkpoints/" + ckpt_name

    #initialize logger
    run_name = cfg.run_name if hasattr(cfg, "run_name") else f"Predict_Diff_{cfg.data.name}_{cfg.style_sampling.name}_cfg{cfg.cfg_scale}"
    logger = pl_loggers.WandbLogger(project="Semantic Style Diffusion", name=run_name)

    # delete pretrained ckpt path
    del cfg.diffusion.ckpt_path
    # load module
    module = LDM_Diffusion.load_from_checkpoint(ckpt_path, cfg=cfg, map_location=torch.device("cpu"), strict=False)
    # module = LDM_Diffusion.load_from_checkpoint(ckpt_path)

    # set float point precision
    torch.set_float32_matmul_precision('high')

    #dataset cfg
    for name in cfg.data.dataset:
        ds_name = name
    ds_config = cfg.data.dataset[ds_name]

    # load an image and predict
    test_folder_path = cfg.location.data_dir + "/" + cfg.data.test_folder

    predict_dir = cfg.location.data_dir + "/syn_data_report/" + f"{cfg.run_name}"
    Path(predict_dir).mkdir(parents=True, exist_ok=True)
    
    # load test condition image
    test_img = np.array(Image.open(test_folder_path + "/test_5.png").convert('L'))

    if cfg.data.name == "cityscapes":
        test_img = (test_img == 26).astype(np.uint8)

    out_ch = 2
    c = F.one_hot(torch.from_numpy(test_img).to(module.device).to(torch.long), num_classes=out_ch).unsqueeze(0).to(torch.float32)

    # load style images
    test_style_path = test_folder_path + "/" + cfg.style_sampling.name

    with torch.no_grad():
        # set to validation
        module._model.eval()

        if cfg.style_sampling.name == "nearby" or cfg.style_sampling.name == "augmented":
            style_0 = (torch.from_numpy(np.array(Image.open(test_style_path + "/1_img.png"))[:,:,:3]).to(torch.float32).to(module.device).unsqueeze(0).unsqueeze(0) / 127.5)-1
        elif cfg.style_sampling.name == "mp":
            style_0 = []
            for i in range(cfg.style_sampling.num_patches):
                style_0.append((torch.from_numpy(np.array(Image.open(test_style_path + f"/1_img_{i}.png"))[:,:,:3]).to(torch.float32).to(module.device).unsqueeze(0).unsqueeze(0) / 127.5)-1)
            style_0 = torch.concat(style_0, dim=1)
        else:
            style_0 = (torch.zeros((1,1,512,512,3), dtype=torch.float32, device=module.device) / 127.5)-1

        ldm_batch_0 = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=module.device), "segmentation": c, "style_imgs":style_0}
        
        # create conditioning tensor
        z, c_0 = module._model.get_input(ldm_batch_0, "image")

        # get uncond conditioning
        ldm_batch_uncond = {"image": torch.zeros((1,512,512,3), dtype=torch.float32, device=module.device), "segmentation": c, "style_imgs":torch.zeros_like(style_0)-2}
        z, c_uncond = module._model.get_input(ldm_batch_uncond, "image")

        style_images = ldm_batch_0["style_imgs"]
        style_features = module._model._agg_block(style_images)

        for itr in range(5):
            #sample images
            img_ddim_w0, _ = module._model.sample_log(c_0, batch_size=1, ddim=True, ddim_steps=cfg.ddim_steps, eta=cfg.eta, log_every_t=1000, unconditional_conditioning=c_uncond, unconditional_guidance_scale=cfg.cfg_scale)

            img_ddim_w0 = torch.clip(module._model.decode_first_stage(img_ddim_w0), -1, 1)

            gen_img_features = module._model._agg_block._embedder(img_ddim_w0)

            cos_sim = F.cosine_similarity(style_features, gen_img_features, dim=-1)
            
            img_ddim_w0 = ((img_ddim_w0[0].permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
            # save img
            Image.fromarray(img_ddim_w0).save(predict_dir + f"/img_0{itr}.png")
            print(f"cosine similarity img_0{itr}: {cos_sim}\n")

        # save seg
        # seg_0 = torch.argmax(ldm_batch_0["segmentation"], dim=-1).cpu().numpy().astype(np.uint8).squeeze(0)
        # Image.fromarray(seg_0).save(predict_dir + f"/seg_00.png")


if __name__ == "__main__":
    main()