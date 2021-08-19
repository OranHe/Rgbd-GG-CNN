import numpy as np

import torch
import torch.utils.data


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self):
        self.grasp_files = []

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_posimg(self, idx):
        raise NotImplementedError()
    def get_angimg(self, idx):
        raise NotImplementedError()
    def get_depth(self, idx):
        raise NotImplementedError()
    def get_rgb(self, idx):
        raise NotImplementedError()

    def __getitem__(self, idx):
        depth_img = self.get_depth(idx)
        rgb_img = self.get_rgb(idx)

        pos_img = self.get_posimg(idx) #二值化圖
        ang_img = self.get_angimg(idx) #np矩陣
        x = self.numpy_to_torch(rgb_img)
        y = self.numpy_to_torch(depth_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2*ang_img))
        cos = torch.reshape(cos, (1, 600, 600))
        sin = self.numpy_to_torch(np.sin(2*ang_img))
        sin = torch.reshape(sin, (1, 600, 600))
        ang = self.numpy_to_torch(ang_img)
        ang = torch.reshape(ang, (1, 600, 600))
        return x, y, (pos, cos, sin), idx, ang

    def __len__(self):
        return len(self.grasp_files)
