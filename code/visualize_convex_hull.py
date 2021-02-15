import os
import torch
import argparse
import numpy as np
import trimesh
from skimage import measure
from scipy.spatial import cKDTree

from core_lib.cameras import PinholeCamera

from utils.general import load_image, load_camera_by_extension, load_point_cloud_by_file_extension, convex_hull
from model.sample import MinGlobalToSurfaceDistance, NormalPerPoint
from visualize_sampler import get_sphere_transform, add_point_to_scene


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--case-path',
        help='Case directory',
        required=True)
    parser.add_argument(
        '--mc-resolution',
        help='Resolution used in marching cubes',
        type=int,
        default=128)
    parser.add_argument(
        '--global-sigma',
        help='Global sigma used by sampler',
        type=float,
        default=1.0)
    parser.add_argument(
        '--max-points',
        help='Max points to be shown',
        type=int,
        default=500)
    parser.add_argument(
        '--point-scale',
        help='Scale of the points',
        type=float,
        default=0.01)

    args = parser.parse_args()

    case_path = args.case_path
    case_views = len([f for f in os.listdir(os.path.join(case_path, 'images')) if f.endswith('.jpg')])

    # Generate convex hull
    # 1. Load data ch data
    masks = [load_image(os.path.join(case_path, 'masks', f'mask_{v}.jpg'), mode='L')/255 for v in range(case_views)]
    cams = [load_camera_by_extension(os.path.join(case_path, 'cameras_preproc', f'camera_{v}.npy')) for v in range(case_views)]

    # 2. Generate sampling points
    x = y = z = np.linspace(-1., 1., args.mc_resolution)
    spacing = x[2] - x[1]
    points = np.stack(np.meshgrid(x, y, z), -1).astype(np.float32).reshape((-1,3))
    points = torch.tensor(points).float()

    # 3. Compute convex hull
    ch = convex_hull(points, cams, masks).numpy()
    occ = 1.0 - 2*ch # {-1: inside, 1: outside}

    # 4. Reconstruct using marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        volume=occ.reshape([args.mc_resolution, args.mc_resolution, args.mc_resolution]).transpose([1,0,2]),
        level=0,
        spacing=(spacing, spacing, spacing))
    verts += np.array([x[0], y[0], z[0]])
    mesh_ch = trimesh.Trimesh(verts, faces)
    mesh_ch.export(os.path.join(case_path, 'convex_hull.obj'))

    # Sample points with global sampler
    mesh = trimesh.load(os.path.join(case_path, 'mesh_preproc.obj'))
    vertices = torch.unsqueeze(torch.from_numpy(mesh.vertices).float(), dim=0)
    kdtree = cKDTree(mesh.vertices)
    sampler = MinGlobalToSurfaceDistance(global_sigma=args.global_sigma, min_glob2surf_dist=0.05)
    p_global = sampler.get_points_global([kdtree], batch_size=1, n_points=args.max_points, dim=3, device=vertices.device).numpy().squeeze()

    # Visualize scene
    scene = trimesh.Scene()
    mesh.visual.face_colors = [128, 128, 128, 90]
    scene.add_geometry(mesh, node_name='mesh')
    for p_g in p_global:
        add_point_to_scene(scene, p_g, scale=args.point_scale, color=[255,0,0,90])
    mesh_ch.visual.face_colors = [0, 255, 0, 30]
    scene.add_geometry(mesh_ch, node_name='ch')

    scene.show()
