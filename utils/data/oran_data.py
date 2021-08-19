import os
import glob
import cv2
import numpy as np
from .grasp_data import GraspDatasetBase

class OranDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(OranDataset, self).__init__()

        graspf = glob.glob(os.path.join(file_path, '*', '*q.jpg'))
        graspf.sort()
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
        else:
            print("Dataset: Load",str(l)," images")
        depthf = [f.replace('q.jpg', 'd.jpg') for f in graspf]
        rgbf = [f.replace('d.jpg', '.jpg') for f in depthf]
        anglef = [f.replace('.jpg', '.npy') for f in rgbf]

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]
        self.angle_files = anglef[int(l * start):int(l * end)]
    def normalize(self,pic):
        pic = pic.astype(np.float32)/255.0
        pic -= pic.mean()
        return pic
    def get_rgb(self, idx):
        rgb_img = self.normalize(cv2.imread(self.rgb_files[idx],-1)).transpose((2, 0, 1))
        return rgb_img
    def get_depth(self, idx):
        depth_img = self.normalize(cv2.imread(self.depth_files[idx],-1)).transpose((2, 0, 1))
        return depth_img
    def get_posimg(self, idx):
        grasp_img = cv2.imread(self.grasp_files[idx],0)/255
        return grasp_img
    def get_angimg(self, idx):
        angle_img = np.load(self.angle_files[idx])
        return angle_img