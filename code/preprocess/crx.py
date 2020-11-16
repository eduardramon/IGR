import os
import sys
import argparse
import json

import trimesh
import numpy as np
from tqdm import tqdm

from core_lib.cameras import PinholeCamera
from core_lib.utilities import create_directory

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)

import utils.general as utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', required=True, type=str, help='Database path')
    parser.add_argument('--skip-walls', action='store_true', help='Skip cases with walls')
    parser.add_argument('--output-dataset-path', required=True, type=str, help='Output dataset path')

    args = parser.parse_args()

    with open(args.dataset_path) as f:
        dataset = json.load(f)

    # Remove cases with background walls
    if args.skip_walls:
        dataset['database']['samples'] = [s for s in dataset['database']['samples'] if not s['background_wall']]

    # Compute stats for normalization
    min_corner = [-1,-1,-1]
    max_corner = [ 1, 1, 1]
    mean_bounds = np.zeros((2,3))
    print('Computing stats for normalization')
    for sample in tqdm(dataset['database']['samples']):
        # Compute mean center
        mean_bounds += np.array(sample['mesh_bounds']) / len(dataset['database']['samples'])
    mean_center = np.sum(mean_bounds, axis=0) / 2
    max_mean_edge = np.max(mean_bounds[1]-mean_bounds[0])
    target_center = (np.array(min_corner) + np.array(max_corner)) / 2
    target_max_edge = np.max(np.array(max_corner) - np.array(min_corner))

    displacement = target_center - mean_center
    scaling = target_max_edge / max_mean_edge

    # Normlaize samples
    print('Normalizing and caching samples')
    for sample in tqdm(dataset['database']['samples']):

        # Load vertices and normals
        mesh_file = sample['mesh']
        mesh = trimesh.load_mesh(mesh_file)

        # Normalize vertices
        point_set_mnlfld = mesh.vertices.astype(np.float32)
        point_set_mnlfld = (point_set_mnlfld + displacement) * scaling

        # Export preprocessed mesh
        mesh_preproc_file = os.path.splitext(mesh_file)[0]+'_preproc.obj'
        trimesh.Trimesh(point_set_mnlfld, mesh.faces).export(mesh_preproc_file)
        sample['mesh_preproc'] = mesh_preproc_file

        # Cache mesh for faster loading
        sample['mesh_preproc_cached'] = f'{mesh_preproc_file}.memmap'
        normals = mesh.vertex_normals.astype(np.float32)
        point_set_mnlfld = np.concatenate((point_set_mnlfld, normals), axis=-1)
        # Save memmap
        fp = np.memmap(sample['mesh_preproc_cached'], dtype='float32', mode='w+', shape=point_set_mnlfld.shape)
        fp[:] = point_set_mnlfld[:]
        del fp

        # Normalize cameras
        cameras_preproc_dir = os.path.join(os.path.dirname(mesh_file), 'cameras_preproc')
        cameras_preproc = []
        cameras_preproc_cached = []
        create_directory(cameras_preproc_dir)
        for idx, c in enumerate(sample['cameras']):
            # Preproc
            cam_preproc_file = os.path.join(cameras_preproc_dir, f'camera_{idx}.json')
            cam_preproc = PinholeCamera().load(c)
            cam_preproc.pose[:3,3] = (cam_preproc.pose[:3,3] + displacement) * scaling
            cam_preproc.inverse_pose = np.linalg.inv(cam_preproc.pose)
            cam_preproc.save(cam_preproc_file)
            cameras_preproc.append(cam_preproc_file)

            # Preproc and cached
            cam_preproc_cached = np.dot(cam_preproc.calibration, cam_preproc.inverse_pose[:3,])
            cam_preproc_cached_file = os.path.join(cameras_preproc_dir, f'camera_{idx}.npy')
            np.save(cam_preproc_cached_file, cam_preproc_cached)
            cameras_preproc_cached.append(cam_preproc_cached_file)

        sample['cameras_preproc'] = cameras_preproc
        sample['cameras_preproc_cached'] = cameras_preproc_cached

    with open(args.output_dataset_path, 'w') as f:
        json.dump(dataset, f, indent=4)

    sys.exit(0)