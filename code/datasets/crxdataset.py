import torch
import torch.utils.data as data
import trimesh
import json
import numpy as np
import os
import tqdm
import utils.general as utils


class CRXDataSet(data.Dataset):

    def __init__(self, dataset_path, split, points_batch=1024, d_in=3, with_gt=False, with_normals=False):

        self.dataset_path = dataset_path
        self.split = split
        self.points_batch = points_batch
        self.d_in = d_in
        self.with_gt = with_gt
        self.with_normals = with_normals

        self.load(dataset_path)
        self.cache(min_corner=[-1,-1,-1], max_corner=[1,1,1])

    def load_points(self, index):
        mesh_file = self.samples[index]['mesh']
        mesh_file_cached = f'{mesh_file}.memmap'
        if os.path.exists(mesh_file_cached):
            return np.memmap(mesh_file_cached, dtype='float32', mode='r').reshape(-1,6).astype(np.float32)
        else:
            return trimesh.load_mesh(mesh_file).vertices.astype(np.float32)

    def __getitem__(self, index):

        point_set_mnlfld = torch.from_numpy(self.load_points(index)).float()

        random_idx = torch.randperm(point_set_mnlfld.shape[0])[:self.points_batch]
        point_set_mnlfld = torch.index_select(point_set_mnlfld, 0, random_idx)

        if self.with_normals:
            normals = point_set_mnlfld[:, -self.d_in:]  # todo adjust to case when we get no sigmas

        else:
            normals = torch.empty(0)

        return point_set_mnlfld[:, :self.d_in], normals, index

    def __len__(self):
        return len(self.samples)

    def get_info(self, index):
        return [self.samples[index]['case_identifier']]

    def load(self, dataset_path):
        with open(dataset_path) as f:
            self.dataset = json.load(f)
        self.samples = [s for s in self.dataset['database']['samples'] if s['case_identifier'] in self.split]

    def cache(self, min_corner=[-1,-1,-1], max_corner=[1,1,1]):

        # Compute normalization stats
        mean_bounds = np.zeros((2,3))
        for sample in self.samples:
            # Compute mean center
            mean_bounds += np.array(sample['mesh_bounds']) / len(self.samples)
        mean_center = np.sum(mean_bounds, axis=0) / 2
        max_mean_edge = np.max(mean_bounds[1]-mean_bounds[0])
        target_center = (np.array(min_corner) + np.array(max_corner)) / 2
        target_max_edge = np.max(np.array(max_corner) - np.array(min_corner))

        displacement = target_center - mean_center
        scaling = target_max_edge / max_mean_edge

        # Cache
        for sample in tqdm.tqdm(self.samples):
            mesh_file = sample['mesh']
            mesh_file_cached = f'{mesh_file}.memmap'
            if os.path.exists(mesh_file_cached): continue

            # Load vertices and normals
            mesh = trimesh.load_mesh(mesh_file)
            point_set_mnlfld = mesh.vertices.astype(np.float32)
            point_set_mnlfld = (point_set_mnlfld + displacement) * scaling
            normals = mesh.vertex_normals.astype(np.float32)
            point_set_mnlfld = np.concatenate((point_set_mnlfld, normals), axis=-1)
            # Save memmap
            fp = np.memmap(mesh_file_cached, dtype='float32', mode='w+', shape=point_set_mnlfld.shape)
            fp[:] = point_set_mnlfld[:]
            del fp
