from glob import glob
import os
import cv2
import torch
from .base_dataset import BaseTsSet

class FolderTestSet(BaseTsSet):
    def __init__(self, img_dir, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms)
        
        # 支援 jpg, png 等格式
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        img_paths = []
        for ext in exts:
            img_paths += glob(os.path.join(img_dir, ext))
        
        # 排序確保順序性
        img_paths = sorted(img_paths)

        # 存入 base class 屬性
        self.img_path_list = img_paths
        self.file_name_list = [os.path.basename(p) for p in img_paths]