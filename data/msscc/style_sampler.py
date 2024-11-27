import torch
import numpy as np
import random

from data.catch.catch_utils import wsi_sample

class NoneSampler:
    def __init__(self, cfg, transforms) -> None:
        self._cfg = cfg
        self._transforms = transforms

    def sample_imgs(self, slide_obj, pos, offset, p_size, b_scale, sample_list, lookup_f):
        return self._transforms(image=np.zeros((p_size, p_size, 3), dtype=np.uint8))["image"].unsqueeze(0)


class NearbySampler:
    def __init__(self, cfg, transforms) -> None:
        self._cfg = cfg
        self._transforms = transforms

    def sample_imgs(self, slide_obj,sample_list):
        patch_coords = random.choice(sample_list)

        style_crop = slide_obj.get_centered_patch(patch_coords[1],patch_coords[0])

        style_crop = self._transforms(image=style_crop)["image"][None,:]
        return style_crop


class MultiPatchSampler:
    def __init__(self, cfg, transforms) -> None:
        self._cfg = cfg
        self._transforms = transforms

        self._num_patches = cfg.num_patches

    def sample_imgs(self, slide_obj,sample_list):
        style_imgs = []
        for i in range(self._num_patches):
            sampled_coords = np.random.randint(0, len(sample_list))
            patch_coords = sample_list[sampled_coords]

            style_crop = slide_obj.get_centered_patch(patch_coords[1],patch_coords[0])

            # apply basic augments
            style_crop = self._transforms(image=style_crop)["image"]
            style_imgs.append(style_crop)

        style_imgs = torch.stack(style_imgs, axis=0)
        return style_imgs