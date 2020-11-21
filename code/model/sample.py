import torch
import utils.general as utils
import abc
import numpy as np


class Sampler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_points(self,pc_input):
        pass

    @staticmethod
    def get_sampler(sampler_type):

        return utils.get_class("model.sample.{0}".format(sampler_type))


class NormalPerPoint(Sampler):

    def __init__(self, global_sigma, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input, local_sigma=None):
        sample_local, sample_global = self.get_points_local_global(pc_input, local_sigma)
        return torch.cat([sample_local, sample_global], dim=1)

    def get_points_local_global(self, pc_input, local_sigma=None):
        batch_size, sample_size, dim = pc_input.shape

        sample_local = self.get_points_local(pc_input, sample_size, local_sigma)
        sample_global = self.get_points_global(pc_input, n_points=sample_size//8)

        return sample_local, sample_global

    def get_points_local(self, pc_input, n_points, local_sigma=None):
        batch_size, sample_size, dim = pc_input.shape

        if local_sigma is not None:
            return pc_input[:,:n_points,:] + (torch.randn(batch_size, n_points, dim, device=pc_input.device) * local_sigma.unsqueeze(-1))
        else:
            return pc_input[:,:n_points,:] + (torch.randn(batch_size, n_points, dim, device=pc_input.device) * self.local_sigma)

    def get_points_global(self, pc_input, n_points):
        batch_size, sample_size, dim = pc_input.shape
        return (torch.rand(batch_size, n_points, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma

class MinGlobalToSurfaceDistance(NormalPerPoint):

    def __init__(self, global_sigma, local_sigma=0.01, min_glob2surf_dist=0.1):
        super().__init__(global_sigma, local_sigma)
        self.min_glob2surf_dist = min_glob2surf_dist

    def get_points(self, pc_input, kdtrees, local_sigma=None):
        sample_local, sample_global = self.get_points_local_global(pc_input, kdtrees, local_sigma)
        return torch.cat([sample_local, sample_global], dim=1)

    def get_points_local_global(self, pc_input, kdtrees, local_sigma=None):
        batch_size, sample_size, dim = pc_input.shape

        sample_local = self.get_points_local(pc_input, sample_size, local_sigma)
        sample_global = self.get_points_global(kdtrees, batch_size, sample_size//8, dim, pc_input.device)

        return sample_local, sample_global

    def get_points_global(self, kdtrees, batch_size, n_points, dim, device):
        return torch.cat([torch.unsqueeze(self.sample_global(kdtrees[b], n_points, dim), dim=0) for b in range(batch_size)], dim=0).to(device)

    def sample_global(self, kdtree, n_points, dim):

        samples = torch.empty((0,dim))
        while len(samples) < n_points:
            candidate_samples = (torch.rand(n_points-len(samples), dim, device='cpu') * (self.global_sigma * 2)) - self.global_sigma
            dd, _ = kdtree.query(candidate_samples, distance_upper_bound=self.min_glob2surf_dist, k=1)
            samples = torch.cat([samples, candidate_samples[dd==np.inf]], dim=0)

        return samples

