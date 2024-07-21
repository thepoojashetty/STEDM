import os
import sys
import cv2
import shutil
import scipy.io

import albumentations as A
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2

from data.cityscapes.cityscapes_ds import Cityscapes_DS,Cityscapes_DS_Predict

from ldm.util import get_obj_from_str
from sklearn.model_selection import train_test_split
import zipfile

class Cityscapes_DM_Anno(pl.LightningDataModule):
    def __init__(self, cfg, ds_cfg, ratio, **kwargs):
        super().__init__()
        # store configs
        self._cfg = cfg
        self._ds_cfg = ds_cfg

        # location based parameters
        self._data_dir = cfg.location.data_dir
        self._n_workers = cfg.location.n_workers
        self._location = cfg.location.name
        # basic data parameters
        self._batch_size = cfg.data.batch_size
        self._patch_size = cfg.data.patch_size
        self._num_classes = cfg.data.num_classes
        # sample number for this dataloader
        self._samples = int(cfg.data.samples * ratio)
        # data storage parameters
        self._reload_data = cfg.data.reload_data
        self._zip_name = ds_cfg.zip_name + "_" + ds_cfg.annotator.name  if hasattr(ds_cfg, "annotator") else ds_cfg.zip_name
        pass


    def prepare_data(self):
        base_dir = self._data_dir + "/" + self._zip_name
        zip_file = self._data_dir + "/" + self._zip_name + ".zip"
        
        # refresh data, load wsi, load annos, create label mask, create zip
        if self._location != "pc" and (self._reload_data or not os.path.isfile(zip_file)):
            # shutil.make_archive(zip_file[:-4], 'zip', base_dir)
            selected_folders = [ os.path.join(base_dir,self._cfg.data.train_clear_images_path,"train"), 
                                 os.path.join(base_dir,self._cfg.data.train_foggy_images_path,"train"),
                                 os.path.join(base_dir,self._cfg.data.label_path,"train")
                            ]
            self.create_zip_of_selected_folders(base_dir,selected_folders, zip_file)
        
        if self._location == "pc":
            # if not os.path.isdir(base_dir):
            #     shutil.unpack_archive(zip_file, base_dir)
            pass
        else:
            local_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            local_zip_file = local_dir + "/" + self._zip_name + ".zip"
            print(f"Copy {self._zip_name} zip file")
            sys.stdout.flush()
            shutil.copyfile(zip_file, local_zip_file)
            print(f"Unpack {self._zip_name} zip file")
            sys.stdout.flush()
            shutil.unpack_archive(local_zip_file, local_dir + "/" + self._zip_name)
            print(f"Delete {self._zip_name} zip file")
            sys.stdout.flush()
            os.remove(local_zip_file)
            print(f"Finished {self._zip_name} preparation")
            sys.stdout.flush()

    def create_zip_of_selected_folders(self,data_dir, selected_folders, output_zip):
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for folder in selected_folders:
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path):
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, data_dir)
                            zipf.write(file_path, arcname)

    def setup(self, stage):
        if self._location == "pc":
            base_dir = self._data_dir + "/" + self._zip_name
        else:
            base_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID']) + "/" + self._zip_name

        # split images category wise
        self._list_train = []
        self._list_val = []
        self._list_test = []
        self._list_unanno = []
            
        self._list_data = self.get_sample_path_list(os.path.join(base_dir,self._cfg.data.train_clear_images_path), os.path.join(base_dir,self._cfg.data.label_path), "train", ".png")
        # self._list_test = self.get_sample_path_list(os.path.join(base_dir,self._cfg.data.test_images_path), os.path.join(base_dir,self._cfg.data.label_path), "val", ".png")
        # self._list_unanno = self.get_sample_path_list(base_dir + "/imgs", base_dir + "/segs", "unanno", ".jpg")

        # setup augments
        base_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                                      A.ToFloat(max_value=255), ToTensorV2()])

        val_test_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.ToFloat(max_value=255), ToTensorV2()])

        style_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                             A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-360,360), shear=(-20,20), mode=cv2.BORDER_REFLECT, p=1.0),
                             A.ToFloat(max_value=255), ToTensorV2()])

        # setup style sampler class
        style_sampler_class = get_obj_from_str("data.cityscapes.style_sampler." + self._cfg.style_sampling.class_name)
        style_sampler = style_sampler_class(self._cfg.style_sampling, style_transforms)
        style_sampler_pred = style_sampler_class(self._cfg.style_sampling, style_transforms)


        # get style drop rate
        style_drop_rate = self._cfg.style_drop_rate if hasattr(self._cfg, "style_drop_rate") else 0.0

        # create datasets
        self._list_train, self._list_unanno = train_test_split(self._list_data, test_size=self._cfg.data.unanno_keep_ratio, random_state=42)
        self._ds_train= Cityscapes_DS(self._list_train, self._samples, self._num_classes, base_transforms, style_sampler, style_drop_rate,self._ds_cfg)
        # self._ds_val = Cityscapes_DS_ValTest(self._list_val, self._num_classes, val_test_transforms, style_sampler, self._ds_cfg)
        # self._ds_test = Cityscapes_DS_ValTest(self._list_test, self._num_classes, val_test_transforms, style_sampler, self._ds_cfg)
        self._ds_predict = Cityscapes_DS_Predict(self._list_train, self._samples, self._num_classes, base_transforms, style_sampler_pred, 0.0)
    
    #returns a list of tuples with the image and segmentation paths
    def get_sample_path_list(self,image_dir_path,seg_dir_path,split,suffix=".png"):
        file_paths = []
        image_split_root = os.path.join(image_dir_path,split)
        seg_split_root = os.path.join(seg_dir_path,split)
        for root, dirs, files in os.walk(image_split_root):
            for file in files:
                if file.endswith(suffix):
                    city_name = root.split("/")[-1]
                    seg_file_name = "_".join(file.split("_")[:3]) + "_gtFine_labelIds.png"
                    sample_path = (os.path.join(root, file),os.path.join(seg_split_root,city_name,seg_file_name))
                    file_paths.append(sample_path)
        return file_paths


    def train_dataloader(self):
        return DataLoader(self._ds_train, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=True, persistent_workers=True)
    
    def train_dataset(self):
        return self._ds_train
  
    def val_dataloader(self):
        return DataLoader(self._ds_val, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=True, persistent_workers=True)
    
    def val_dataset(self):
        return self._ds_val

    def test_dataloader(self):
        return DataLoader(self._ds_test, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False, persistent_workers=True)
    
    def test_dataset(self):
        return self._ds_test

    def predict_dataloader(self):
        return DataLoader(self._ds_predict, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False, persistent_workers=True)

    def predict_dataset(self):
        return self._ds_predict


class Cityscapes_DM_UnAnno(Cityscapes_DM_Anno):
        def setup(self, stage):
            if self._location == "pc":
                base_dir = self._data_dir + "/" + self._zip_name
            else:
                base_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID']) + "/" + self._zip_name

            # split images category wise
            self._list_train = []
            self._list_val = []
            self._list_test = []
            self._list_unanno = []

            self._list_data = self.get_sample_path_list(os.path.join(base_dir,self._cfg.data.train_clear_images_path), os.path.join(base_dir,self._cfg.data.label_path), "train", ".png")
            
            # setup augments
            base_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                                        A.ToFloat(max_value=255), ToTensorV2()])

            style_transforms = A.Compose([A.Resize(self._patch_size, self._patch_size), A.HorizontalFlip(), A.VerticalFlip(),
                                A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-360,360), shear=(-20,20), mode=cv2.BORDER_REFLECT),
                                A.ToFloat(max_value=255), ToTensorV2()])

            # setup style sampler class
            style_sampler_class = get_obj_from_str("data.flowers.style_sampler." + self._cfg.style_sampling.class_name)
            style_sampler_pred = style_sampler_class(self._cfg.style_sampling, style_transforms)

            self._list_train, self._list_unanno = train_test_split(self._list_data, test_size=self._cfg.data.unanno_keep_ratio, random_state=42)
            self._list_unanno = self.get_unanno_path_list(self._list_unanno,base_dir)

            # create datasets
            self._ds_train = []
            self._ds_val = []
            self._ds_test = []
            self._ds_predict = Cityscapes_DS(self._list_unanno, self._samples, self._num_classes, base_transforms, style_sampler_pred, 0.0)

        def get_unanno_path_list(self,unanno_list,base_dir):
            unanno_new_list = []
            for image_path,label_path in unanno_list:
                image_name = image_path.split("/")[-1]
                unanno_image_name = "_".join(image_name.split("_")[:3]) + "_leftImg8bit_foggy_beta_0.01.png"
                city_name = image_path.split("/")[-2]
                unanno_image_path = os.path.join(base_dir,self._cfg.data.train_foggy_images_path,city_name,unanno_image_name)
                unanno_new_list.append((unanno_image_path,label_path))
            return unanno_new_list