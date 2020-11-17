import os

import numpy as np
import torch
import trimesh
from PIL import Image

from core_lib.cameras import PinholeCamera

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj


def load_point_cloud_by_file_extension(file_name):

    ext = file_name.split('.')[-1]

    if ext == "npz" or ext == "npy":
        point_set = torch.tensor(np.load(file_name)).float()
    else:
        point_set = torch.tensor(trimesh.load(file_name, ext).vertices).float()

    return point_set


def load_camera_by_extension(file_name):

    ext = file_name.split('.')[-1]

    if ext == "npy":
        return torch.tensor(np.load(file_name)).float()
    elif ext == "json":
        cam = PinholeCamera().load(file_name)
        cam_proj = np.dot(cam.calibration, cam.inverse_pose[:3,])
        return torch.tensor(cam_proj).float()
    else:
        raise ValueError(f"Unsupported extension for cameras {ext}")


def load_image(file_name, mode='RGB'):

    img = Image.open(file_name).convert(mode=mode)
    return torch.tensor(np.array(img)).float()


def project(points, camera):

    P, D = points.size()

    point_h = torch.cat((points, torch.ones((P,1))), dim=-1)
    point_h_proj = torch.mm(camera, point_h.T).T

    return point_h_proj[:, :2] / point_h_proj[:, -1:]


def bilinear_sampling(im, points):

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    dtype_long = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    x, y = points[:,0], points[:,1]

    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
    wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
    wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
    wd = (x-x0.type(dtype)) * (y-y0.type(dtype))

    return torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)


def convex_hull(points, cams, masks, s=100):

    # Soft mask values are assumed to be normalized between 0 and 1.

    P, D = points.size()
    sigmoid = torch.nn.Sigmoid()

    convex_hull = torch.ones(P)
    for mask, cam in zip(masks, cams):

        p2d = project(points, cam)
        v = bilinear_sampling(mask, p2d)

        convex_hull *= sigmoid(s*(v-0.5))

    return convex_hull


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)