import torch
import torch.utils.data as data
import trimesh
import json
import numpy as np
import os
import tqdm
import utils.general as utils
import random


class CRXDataSet(data.Dataset):

    def __init__(self, dataset_path, split, points_batch=1024, d_in=3, with_gt=False, with_normals=False, with_symmetrics=False):

        self.dataset_path = dataset_path
        self.split = split
        self.points_batch = points_batch
        self.d_in = d_in
        self.with_gt = with_gt
        self.with_normals = with_normals
        self.with_symmetrics = with_symmetrics

        self.load(dataset_path)

    def load_points_normals(self, index, sym):
        identifier = self.identifiers[index] + ('_sym' if sym else '')
        index = self.map[identifier]
        points = np.memmap(self.samples[index]['mesh_preproc_cached'], dtype='float32', mode='r').reshape(-1,6).astype(np.float32)

        # This is for an experimental purpose
        keep_left = bool(index % 2) 
        if keep_left:
            half_idx = points[:,0] > -0.1
        else:
            half_idx = points[:,0] < 0.1

        return points[half_idx]

    def __getitem__(self, index):

        is_sym = bool(random.getrandbits(1)) if self.with_symmetrics else False
        point_set_mnlfld = torch.from_numpy(self.load_points_normals(index, is_sym)).float()

        random_idx = torch.randperm(point_set_mnlfld.shape[0])[:self.points_batch]
        point_set_mnlfld = torch.index_select(point_set_mnlfld, 0, random_idx)

        if self.with_normals:
            normals = point_set_mnlfld[:, -self.d_in:]  # todo adjust to case when we get no sigmas

        else:
            normals = torch.empty(0)

        return point_set_mnlfld[:, :self.d_in], normals, index, is_sym

    def __len__(self):
        return len(self.identifiers)

    def get_info(self, index):
        return [self.identifiers[index]]

    def load(self, dataset_path):
        with open(dataset_path) as f:
            self.dataset = json.load(f)

        # Get all samples from split
        self.samples = [s for s in self.dataset['database']['samples'] if s['case_identifier'] in self.split]

        # Get samples identifiers from split without symmetrics
        self.identifiers = [s['case_identifier'] for s in self.samples if '_sym' not in s['case_identifier']]

        # Identifiers-Index map
        self.map = {s['case_identifier']:i for i, s in enumerate(self.samples)}
