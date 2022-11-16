import numpy as np
import open3d as o3d
import torch
import random
import os
import sys
import errno
import os.path as osp

def normalize(points, scale=1.0):
    centroid = np.mean(points, axis=0)
    points_new = points - centroid
    m = np.max(np.sqrt(np.sum(points_new ** 2, axis=1)))
    points_new = scale * points_new / m
    return points_new


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return


def read_point_cloud_ply(file_path, require_normal=False):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if not require_normal:
        return points, colors
    else:
        normals = np.asarray(pcd.normals)
        return points, colors, normals


def draw_point_cloud(points, colors, visualize=True, save_path=''):
    points = o3d.utility.Vector3dVector(points)
    colors = o3d.utility.Vector3dVector(colors)
    pcd = o3d.geometry.PointCloud(points=points)
    pcd.colors = colors
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    if save_path != '':
        o3d.io.write_point_cloud(save_path, pcd)
    return


def read_mesh_ply(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    faces = np.asarray(mesh.triangles)
    return points, colors, faces


def draw_mesh(points, colors, faces, visualize=True, save_path=''):
    points = o3d.utility.Vector3dVector(points)
    colors = o3d.utility.Vector3dVector(colors)
    faces = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(vertices=points, triangles=faces)
    mesh.vertex_colors = colors
    if visualize:
        o3d.visualization.draw_geometries([mesh])
    if save_path != '':
        o3d.io.write_triangle_mesh(save_path, mesh)
    return




def display_config(args):
    print('-------SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" % (str(arg), str(getattr(args, arg))))
    print('')


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
