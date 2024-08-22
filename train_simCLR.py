import hydra
import torch

from data.dm import DataModule
from ssl_training.simCLR import SimCLR

import pytorch_lightning as pl
from datetime import timedelta
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy


@hydra.main(version_base=None, config_path="conf", config_name="config_simCLR")
def main(cfg : DictConfig):
    pl.seed_everything(42)
    #Check if you have GPU
    acceleratoor = "gpu" if torch.cuda.is_available() else "cpu"

    # calculate batch_size
    # cfg.data.batch_size = int(cfg.data.batch_base * cfg.location.batch_mul)
    cfg.data.batch_size = cfg.batch_size

    # calculate learning rate
    # cfg.lr = cfg.base_lr * cfg.data.batch_size * cfg.location.n_gpus

    run_name = cfg.run_name if hasattr(cfg, "run_name") else f"SimCLR_{cfg.data.name}"
    logger = pl_loggers.WandbLogger(project="Semantic Style Diffusion", name=run_name)

    data_module = DataModule(cfg)

    if hasattr(cfg, "ckpt_name"):
        ckpt_name = cfg.ckpt_name
        ckpt_path = cfg.location.result_dir + "/checkpoints/" + ckpt_name
        module = SimCLR.load_from_checkpoint(ckpt_path)
    else:
        module = SimCLR(cfg)

    metric_checkpoint = ModelCheckpoint(dirpath=cfg.location.result_dir + "/checkpoints",
                                       filename=run_name + "_last",
                                       verbose=True, monitor="val_acc_top5", mode="max")
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    progress_bar = TQDMProgressBar(refresh_rate=int(128/cfg.batch_size))
    callbacks = [lr_monitor,  progress_bar, metric_checkpoint] # more callbacks can be added

    trainer = pl.Trainer(max_epochs=cfg.num_epochs,
                         callbacks=callbacks, logger=logger,
                         accelerator=acceleratoor, devices=cfg.location.n_gpus,
                         strategy=DDPStrategy(find_unused_parameters=False, process_group_backend=cfg.location.backend, timeout=timedelta(seconds=7200*4))
                         )

    trainer.fit(module, data_module)


if __name__ == "__main__":
    main()