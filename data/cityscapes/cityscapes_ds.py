import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image


class Cityscapes_DS(torch.utils.data.Dataset):
    def __init__(self, img_l, samples, num_classes, base_transforms, style_sampler, style_drop_rate,ds_cfg):
        self._img_l = img_l
        self._samples = samples
        self._num_classes = num_classes
        self._base_transforms = base_transforms
        self._style_sampler = style_sampler
        self._style_drop_rate = style_drop_rate
        self._ds_cfg = ds_cfg
        self._background_class_value = 0
        self._class_map = dict(zip(ds_cfg.valid_classes, range(1,num_classes)))

    def __len__(self):
        # if len(self._img_l) > 0:
        #     return self._samples
        # else:
        #     return 0
        return len(self._img_l)

    def __getitem__(self, idx):
        # get requested image
        img_tup = self._img_l[idx]

        # load image
        img_raw = np.array(Image.open(img_tup[0]))
        
        # load and process segmenation
        seg = np.array(Image.open(img_tup[1]))

        # map void classes to background
        for cls in self._ds_cfg.void_classes:
            seg[seg == cls] = self._background_class_value
        for cls in self._ds_cfg.valid_classes:
            seg[seg == cls] = self._class_map[cls]

        seg = seg.astype(np.uint8)

        # apply augmentations
        applied = self._base_transforms(image=img_raw, mask=seg)
        img = applied['image']
        seg = applied['mask']

        # get one_hot segmenation
        one_hot = F.one_hot(seg.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)

        style_imgs = self._style_sampler.sample_imgs(img_raw)

        # randomly drop style
        if np.random.uniform(0, 1.0) < self._style_drop_rate:
            style_imgs = torch.zeros_like(style_imgs)-0.5

        # adjust data range and return
        return img*2-1, one_hot, seg, style_imgs*2-1
    

class Cityscapes_DS_ValTest(torch.utils.data.Dataset):
    def __init__(self, img_l, num_classes, base_transforms, style_sampler, ds_cfg):
        self._img_l = img_l
        self._num_classes = num_classes
        self._base_transforms = base_transforms
        self._style_sampler = style_sampler
        self._ds_cfg = ds_cfg
        self._background_class_value = 0
        self._class_map = dict(zip(ds_cfg.valid_classes, range(1,num_classes)))

    def __len__(self):
        return len(self._img_l)

    def __getitem__(self, idx):
        # get requested image
        img_tup = self._img_l[idx]

        # load image
        img_raw = np.array(Image.open(img_tup[0]))
        
        # load and process segmenation
        seg = np.array(Image.open(img_tup[1]))

        # map void classes to background
        for cls in self._ds_cfg.void_classes:
            seg[seg == cls] = self._background_class_value
        for cls in self._ds_cfg.valid_classes:
            seg[seg == cls] = self._class_map[cls]

        seg = seg.astype(np.uint8)

        # apply augmentations
        applied = self._base_transforms(image=img_raw, mask=seg)
        img = applied['image']
        seg = applied['mask']

        # get one_hot segmenation
        one_hot = F.one_hot(seg.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)

        style_imgs = self._style_sampler.sample_imgs(img_raw)

        # adjust data range and return
        return img*2-1, one_hot, seg, style_imgs*2-1


class Cityscapes_DS_Predict(Cityscapes_DS):
    def __getitem__(self, idx):
        return *super().__getitem__(idx), idx
