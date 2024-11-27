import torch

import numpy as np
import torch.nn.functional as F
import pandas as pd
from data.msscc.slide_container import SlideContainer
import os
import glob
from pathlib import Path

from PIL import Image

np.random.seed(42)

class MSSCC_DS(torch.utils.data.Dataset):
    def __init__(self, base_dir, samples, num_classes,base_transforms, style_sampler, style_drop_rate, cfg,predict=False):
        self._base_dir = base_dir
        self._samples = samples
        self._num_classes = num_classes
        self._base_transforms = base_transforms
        self._style_sampler = style_sampler
        self._style_drop_rate = style_drop_rate
        self._patches_per_slide = cfg.data.patches_per_slide
        self._patch_size = cfg.data.patch_size
        self._excluded = cfg.data.nclass.excluded

        self._annotation_file = os.path.join(self._base_dir, cfg.location.msscc.annotation_file)
        self._label_dict = cfg.data.nclass.label_dict
        self._down_factor = cfg.data.down_factor

        self._img_l= self.load_train_files(cfg,self._base_dir,predict)

    def __len__(self):
        if len(self._img_l) > 0:
            return self._samples
        else:
            return 0

    def __getitem__(self, idx):
        sample_idx = np.random.randint(0, len(self._img_l))
        # get requested image
        img_file = self._img_l[sample_idx]
        slide = SlideContainer(img_file, self._annotation_file, self._down_factor, self._patch_size, self._patch_size, label_dict = self._label_dict)

        img_raw, seg, coords = self.get_sample(slide,excluded=self._excluded)

        # apply augmentations
        applied = self._base_transforms(image=img_raw, mask=seg)
        img = applied['image']
        seg = applied['mask']

        seg_adjust = seg.clone()
        seg_adjust[seg_adjust == self._excluded] = 0
        # get one_hot segmenation
        one_hot = F.one_hot(seg_adjust.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)
        
        one_hot[0][seg==self._excluded] = 0

        style_imgs = self._style_sampler.sample_imgs(slide,coords)

        # randomly drop style
        if np.random.uniform(0, 1.0) < self._style_drop_rate:
            style_imgs = torch.zeros_like(style_imgs)-0.5

        # adjust data range and return
        return img*2-1, one_hot, seg, style_imgs*2-1
    
    def get_sample(self,slide_container, excluded=0):
        xmin, ymin = slide_container.get_new_train_coordinates()
        patch = slide_container.get_patch(xmin, ymin)
        y_patch = slide_container.get_y_patch(xmin, ymin, excluded)
        coords = np.argwhere(y_patch > 0)
        if len(coords) == 0:
            coords = np.argwhere(y_patch == 0) # (row,col) -> (y,x) top-left corner
        # element wise addition of xmin and ymin to coords
        coords = coords + [ymin, xmin]
        return (patch, y_patch, coords)
    
    def load_train_files(self,cfg,base_dir,predict=False):
        train_files = []
        csv_path = cfg.location.msscc.data_split_csv_path

        slides = pd.read_csv(csv_path, delimiter=";")

        if predict:
            for index, row in slides.iterrows():
                if row["Dataset"] == "train" and row["Annotation"] == "no":
                    image_file = Path(glob.glob("{}/cs2/{}_cs2.svs".format(base_dir,row["Slide"]), recursive=True)[0])
                    train_files.append(image_file)
        else:
            for index, row in slides.iterrows():
                if row["Dataset"] == "train" and row["Annotation"] == "yes":
                    image_file = Path(glob.glob("{}/p1000/{}_p1000.mrxs".format(base_dir,row["Slide"]), recursive=True)[0])
                    train_files.append(image_file)
        return train_files
    
class MSSCC_DS_Predict(MSSCC_DS):
    def __getitem__(self, idx):
        return *super().__getitem__(idx), idx

class MSSCC_DS_SSL(torch.utils.data.Dataset):
    def __init__(self, base_dir, samples, num_classes,base_transforms, style_sampler, style_drop_rate, cfg):
        self._base_dir = base_dir
        self._samples = samples
        self._num_classes = num_classes
        self._base_transforms = base_transforms
        self._style_sampler = style_sampler
        self._style_drop_rate = style_drop_rate
        self._patches_per_slide = cfg.data.patches_per_slide
        self._patch_size = cfg.data.patch_size
        self._excluded = cfg.data.nclass.excluded

        self._annotation_file = os.path.join(self._base_dir, cfg.location.msscc.annotation_file)
        self._label_dict = cfg.data.nclass.label_dict
        self._down_factor = cfg.data.down_factor

        self._img_l= self.load_train_files(cfg,self._base_dir)

    def __len__(self):
        if len(self._img_l) > 0:
            return self._samples
        else:
            return 0

    def __getitem__(self, idx):
        sample_idx = np.random.randint(0, len(self._img_l))
        # get requested image
        img_file = self._img_l[sample_idx]
        slide = SlideContainer(img_file, self._annotation_file, self._down_factor, self._patch_size, self._patch_size, label_dict = self._label_dict)

        img_raw, seg, coords = self.get_sample(slide,excluded=self._excluded)

        # apply augmentations
        applied = self._base_transforms(image=img_raw, mask=seg)
        img = applied['image']
        seg = applied['mask']

        seg_adjust = seg.clone()
        seg_adjust[seg_adjust == self._excluded] = 0
        # get one_hot segmenation
        one_hot = F.one_hot(seg_adjust.to(torch.long), num_classes=self._num_classes).permute(2,0,1).to(torch.float32)
        
        one_hot[0][seg==self._excluded] = 0

        style_imgs = self._style_sampler.sample_imgs(slide,coords)

        # randomly drop style
        if np.random.uniform(0, 1.0) < self._style_drop_rate:
            style_imgs = torch.zeros_like(style_imgs)-0.5

        dummy_seg = torch.zeros_like(one_hot)

        # adjust data range and return
        return img*2-1, dummy_seg, seg, style_imgs*2-1
    
    def get_sample(self,slide_container, excluded=0):
        xmin, ymin = slide_container.get_new_train_coordinates()
        patch = slide_container.get_patch(xmin, ymin)
        y_patch = slide_container.get_y_patch(xmin, ymin, excluded)
        coords = np.argwhere(y_patch > 0)
        if len(coords) == 0:
            coords = np.argwhere(y_patch == 0) # (row,col) -> (y,x) top-left corner
        # element wise addition of xmin and ymin to coords
        coords = coords + [ymin, xmin]
        return (patch, y_patch, coords)
    
    def load_train_files(self,cfg,base_dir):
        train_files = []
        csv_path = cfg.location.msscc.data_split_csv_path

        slides = pd.read_csv(csv_path, delimiter=";")
        for index, row in slides.iterrows():
            if row["Dataset"] == "train" and row["Annotation"] == "yes":
                image_file = Path(glob.glob("{}/p1000/{}_p1000.mrxs".format(base_dir,row["Slide"]), recursive=True)[0])
                train_files.append(image_file)
            if row["Dataset"] == "train" and row["Annotation"] == "no":
                image_file = Path(glob.glob("{}/cs2/{}_cs2.svs".format(base_dir,row["Slide"]), recursive=True)[0])
                train_files.append(image_file)
        return train_files

class MSSCC_DS_SimCLR(torch.utils.data.Dataset):
    def __init__(self, base_dir, samples, base_transforms, cfg, ds_cfg, train=True):
        self._base_dir = base_dir
        self._samples = samples
        self._base_transforms = base_transforms

        self._patch_size = cfg.data.patch_size
        self._annotation_file = os.path.join(self._base_dir, cfg.location.msscc.annotation_file)
        self._label_dict = cfg.data.nclass.label_dict
        self._down_factor = cfg.data.down_factor
        self._excluded = cfg.data.nclass.excluded

        self._img_l = []
        if train:
            self._img_l= self.load_train_files(cfg,self._base_dir)
        else:
            self._img_l= self.load_val_files(cfg,self._base_dir)
        self.n_views = ds_cfg.n_views

    def __len__(self):
        if len(self._img_l) > 0:
            return self._samples
        else:
            return 0
    
    def __getitem__(self, idx):
        sample_idx = np.random.randint(0, len(self._img_l))
        # get requested image
        img_file = self._img_l[sample_idx]
        slide = SlideContainer(img_file, self._annotation_file, self._down_factor, self._patch_size, self._patch_size, label_dict = self._label_dict)

        img_raw = self.get_image(slide)

        images = [self._base_transforms(image=img_raw)["image"] for _ in range(self.n_views)]
        return images
        
    
    def get_image(self,slide_container):
        xmin, ymin = slide_container.get_new_train_coordinates()
        patch = slide_container.get_patch(xmin, ymin)
        return patch
    
    def load_train_files(self,cfg,base_dir):
        train_files = []
        csv_path = cfg.location.msscc.data_split_csv_path

        slides = pd.read_csv(csv_path, delimiter=";")
        for index, row in slides.iterrows():
            if row["Dataset"] == "train" and row["Annotation"] == "yes":
                image_file = Path(glob.glob("{}/p1000/{}_p1000.mrxs".format(base_dir,row["Slide"]), recursive=True)[0])
                train_files.append(image_file)
            if row["Dataset"] == "train" and row["Annotation"] == "no":
                image_file = Path(glob.glob("{}/cs2/{}_cs2.svs".format(base_dir,row["Slide"]), recursive=True)[0])
                train_files.append(image_file)
        return train_files
    
    def load_val_files(self,cfg,base_dir):
        val_files = []
        csv_path = cfg.location.msscc.data_split_csv_path

        slides = pd.read_csv(csv_path, delimiter=";")
        for index, row in slides.iterrows():
            if row["Dataset"] == "val":
                image_file = Path(glob.glob("{}/cs2/{}_cs2.svs".format(base_dir,row["Slide"]), recursive=True)[0])
                val_files.append(image_file)
        return val_files
    

#second simCLR dataset for augmented view from the same WSI
class MSSCC_DS_SimCLR_2(torch.utils.data.Dataset):
    def __init__(self, base_dir, samples, base_transforms, cfg, ds_cfg, train=True):
        self._base_dir = base_dir
        self._samples = samples
        self._base_transforms = base_transforms

        self._patch_size = cfg.data.patch_size
        self._annotation_file = os.path.join(self._base_dir, cfg.location.msscc.annotation_file)
        self._label_dict = cfg.data.nclass.label_dict
        self._down_factor = cfg.data.down_factor
        self._excluded = cfg.data.nclass.excluded

        self._img_l = []
        if train:
            self._img_l= self.load_train_files(cfg,self._base_dir)
        else:
            self._img_l= self.load_val_files(cfg,self._base_dir)
        self.n_views = ds_cfg.n_views

    def __len__(self):
        if len(self._img_l) > 0:
            return self._samples
        else:
            return 0
    
    def __getitem__(self, idx):
        sample_idx = np.random.randint(0, len(self._img_l))
        # get requested image
        img_file = self._img_l[sample_idx]
        slide = SlideContainer(img_file, self._annotation_file, self._down_factor, self._patch_size, self._patch_size, label_dict = self._label_dict)

        images=[]

        for _ in range(self.n_views):
            img_raw = self.get_image(slide)
            images.append(self._base_transforms(image=img_raw)["image"])

        return images
        
    
    def get_image(self,slide_container):
        xmin, ymin = slide_container.get_new_train_coordinates()
        patch = slide_container.get_patch(xmin, ymin)
        return patch
    
    def load_train_files(self,cfg,base_dir):
        train_files = []
        csv_path = cfg.location.msscc.data_split_csv_path

        slides = pd.read_csv(csv_path, delimiter=";")
        for index, row in slides.iterrows():
            if row["Dataset"] == "train" and row["Annotation"] == "yes":
                image_file = Path(glob.glob("{}/p1000/{}_p1000.mrxs".format(base_dir,row["Slide"]), recursive=True)[0])
                train_files.append(image_file)
            if row["Dataset"] == "train" and row["Annotation"] == "no":
                image_file = Path(glob.glob("{}/cs2/{}_cs2.svs".format(base_dir,row["Slide"]), recursive=True)[0])
                train_files.append(image_file)
        return train_files
    
    def load_val_files(self,cfg,base_dir):
        val_files = []
        csv_path = cfg.location.msscc.data_split_csv_path

        slides = pd.read_csv(csv_path, delimiter=";")
        for index, row in slides.iterrows():
            if row["Dataset"] == "val":
                image_file = Path(glob.glob("{}/cs2/{}_cs2.svs".format(base_dir,row["Slide"]), recursive=True)[0])
                val_files.append(image_file)
        return val_files



    