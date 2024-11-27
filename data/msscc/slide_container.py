import cv2
from shapely import geometry
from pathlib import Path
import json
import numpy as np
import random
import openslide

class SlideContainer:
    def __init__(self, file: Path,
                 annotation_file,
                 down_factor: int = 0,
                 width: int = 256, height: int = 256,
                 sample_func = None, label_dict=None):
        self.file = file
        with open(annotation_file) as f:
            data = json.load(f)
            self.tissue_classes = dict(zip([cat["name"] for cat in data["categories"]],[cat["id"] for cat in data["categories"]]))
            image_id = [i["id"] for i in data["images"] if i["file_name"] == file.name][0]
            self.polygons = [anno for anno in data['annotations'] if anno["image_id"] == image_id]
        self.labels = set([poly["category_id"] for poly in self.polygons])

        # Exclude noisy annotation classes from labels
        self.labels.discard(self.tissue_classes["Bone"])
        self.labels.discard(self.tissue_classes["Cartilage"])
        self.labels.discard(self.tissue_classes["Inflamm/Necrosis"])

        # Initialize sampling probabilities
        # Tumor and non-tumor and each sampled with 45% and within non-tumor classes, each class is sampled uniformly
        # self.probabilities = dict.fromkeys(list(self.labels), np.nan_to_num(np.true_divide(0.45, len(list(self.labels)) - 1), posinf=0))
        if len(self.labels) > 1:
            initial_value = np.nan_to_num(np.true_divide(0.45, len(self.labels) - 1), posinf=0)
        else:
            initial_value = 0.45  # Handle the case where there's only one label

        self.probabilities = dict.fromkeys(self.labels, initial_value)

        self.probabilities.update({max(self.labels): 0.45})
        # Background class is sampled with 10%
        self.probabilities.update({0: 0.1})
        
        self.slide = openslide.open_slide(str(file))

        #compute level
        self.down_factor=down_factor
        self._level =  np.where(np.abs(np.array(self.slide.level_downsamples)-self.down_factor) < 1)[0][0]
        # Background detection using Otsu thresholding
        self.thumbnail = np.array(self.slide.read_region((0, 0), self._level, self.slide.level_dimensions[self._level]))[:, :,:3]
        grayscale = cv2.cvtColor(self.thumbnail, cv2.COLOR_RGB2GRAY)
        grayscale[grayscale == 0] = 255
        blurred = cv2.GaussianBlur(grayscale,(5,5),0)
        self.white, self.mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Find tissue bounding box
        self.x_min, self.y_min = np.min(
            np.array([np.min(np.array(polygon['segmentation']).reshape((-1, 2)), axis=0) for polygon in self.polygons]),
            axis=0)
        self.x_max, self.y_max = np.max(
            np.array([np.max(np.array(polygon['segmentation']).reshape((-1, 2)), axis=0) for polygon in self.polygons]),
            axis=0)
        self.width = width
        self.height = height
        # self.down_factor = self.slide.level_downsamples[level]

        if self._level is None:
            self._level = self.slide.level_count - 1
        self.sample_func = sample_func
        self.label_dict = label_dict

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self.down_factor = self.slide.level_downsamples[value]
        self._level = value

    @property
    def shape(self):
        return self.width, self.height

    @property
    def slide_shape(self):
        return self.slide.level_dimensions[self._level]

    def get_new_level(self):
        return self._level

    def get_patch(self, x: int = 0, y: int = 0):
        patch = np.array(self.slide.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                               level=self._level, size=(self.width, self.height)))
        # Some scanners use 0 in 4th dimension to indicate background -> fill with white
        patch[patch[:, :, -1] == 0] = [255, 255, 255, 0]
        return patch[:,:,:3]


    def get_y_patch(self, x: int = 0, y: int = 0, excluded_val=0):
        y_patch = -1*np.ones(shape=(self.height, self.width), dtype=np.int8)
        inv_map = {v: k for k, v in self.tissue_classes.items()}

        for poly in self.polygons:
            coordinates = np.array(poly['segmentation']).reshape((-1,2))/ self.down_factor
            coordinates = coordinates - (x, y)
            label = self.label_dict[inv_map[poly["category_id"]]]
            #set the colour of the mask
            cv2.drawContours(y_patch, [coordinates.reshape((-1, 1, 2)).astype(int)], -1, label, -1)

        white_mask = cv2.cvtColor(self.get_patch(x,y),cv2.COLOR_RGB2GRAY) > self.white
        excluded = (y_patch == -1)
        y_patch[np.logical_and(white_mask, excluded)] = 0
        #to set all -1 values to 0 in binary segmentation else set it as 3
        y_patch[y_patch == -1] = excluded_val
        return y_patch


    def get_new_train_coordinates(self):
        inv_map = {v: k for k, v in self.tissue_classes.items()}
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.polygons, **{"classes":self.labels ,"size": (self.width, self.height),
                                               "level_dimensions": self.slide.level_dimensions,
                                               "level": self._level})
        # Sample class label according to class probabilities
        label = random.choices(list(self.probabilities.keys()), list(self.probabilities.values()))[0]

        # If background class is sampled, either sample above or below tissue bounding box
        # or right or left from tissue bounding box
        if label == 0:
            if random.choice([True, False]):
                xmin = random.choice([self.x_min - self.width * self.down_factor, self.x_max]) // self.down_factor
                ymin = random.randint(0, self.slide.dimensions[1]) // self.down_factor
            else:
                xmin = random.randint(0, self.slide.dimensions[0]) // self.down_factor
                ymin = random.choice([self.y_min - self.width * self.down_factor, self.y_max]) // self.down_factor

        # For non-background classes, first randomly sample polyon and then randomly sample point within polygon
        else:
            # default sampling method
            xmin, ymin = 0, 0
            found = False
            while not found:
                iter = 0
                polygon = np.random.choice([poly for poly in self.polygons if poly["category_id"] == label])
                coordinates = np.array(polygon['segmentation']).reshape((-1, 2))
                minx, miny, xrange, yrange = polygon["bbox"]
                while iter < 25 and not found:
                    iter += 1
                    pnt = geometry.Point(np.random.uniform(minx, minx + xrange), np.random.uniform(miny, miny + yrange))
                    if geometry.Polygon(coordinates).contains(pnt):
                        xmin = pnt.x // self.down_factor - self.width / 2
                        ymin = pnt.y // self.down_factor - self.height / 2
                        found = True
        return xmin, ymin
    
    #given the center coordinates of the patch, return valid top left coordinates
    def get_centered_patch(self, x_center: int, y_center: int):
        # Calculate the top-left corner of the patch
        x_top_left = int(x_center - self.width // 2)
        y_top_left = int(y_center - self.height // 2)

        # Ensure the top-left corner is within valid bounds
        x_top_left = max(0, x_top_left)
        y_top_left = max(0, y_top_left)

        patch = self.get_patch(x_top_left, y_top_left)
        return patch


    def __str__(self):
        return str(self.file)
