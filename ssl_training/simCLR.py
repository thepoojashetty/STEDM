#reference from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html

import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimCLR(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self._cfg = cfg
        self._hidden_dim = cfg.hidden_dim
        self._lr = cfg.lr
        self._temperature = cfg.temperature
        self._weight_decay = cfg.weight_decay
        self._num_epochs = cfg.num_epochs

        assert self._temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        self.convnet = nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])  # Remove final FC layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 2*self._hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*self._hidden_dim, self._hidden_dim)
        )

    def forward(self, x):
        emb=self.convnet(x)
        emb = emb.view(emb.size(0), -1)
        return emb

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self._lr,
                                weight_decay=self._weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self._num_epochs,
                                                            eta_min=self._lr/50)
        return [optimizer], [lr_scheduler]

    #NXent loss
    def info_nce_loss(self, batch, mode='train'):
        imgs = batch
        imgs = torch.cat(imgs)

        # Encode all images
        feats_f = self.convnet(imgs)
        # Project features
        feats = self.projection(feats_f)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> 2*batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self._temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode+'_loss', nll, on_epoch=True)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean(), on_epoch=True)
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean(), on_epoch=True)
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean(), on_epoch=True)

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')