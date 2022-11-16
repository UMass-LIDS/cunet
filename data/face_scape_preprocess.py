import sys
sys.path.append("..")
from utils import draw_mesh, read_mesh_ply, draw_point_cloud, read_point_cloud_ply
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import random
import trimesh
from data.quantize import quantize_point_cloud
import argparse


expressions = ["1_neutral", "2_smile", "3_mouth_stretch", "4_anger", "5_jaw_left",
               "6_jaw_right", "7_jaw_forward", "8_mouth_left", "9_mouth_right", "10_dimpler",
               "11_chin_raiser", "12_lip_puckerer", "13_lip_funneler", "14_sadness", "15_lip_roll",
               "16_grin", "17_cheek_blowing", "18_eye_closed", "19_brow_raiser", "20_brow_lower"]


def read_mesh_obj(file_path):
    mesh = trimesh.load_mesh(file_path)
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    uvs = np.asarray(mesh.visual.uv)
    text_path = file_path[:-3] + "jpg"
    text = Image.open(text_path)
    text = np.array(text).astype(np.float64) / 255.
    return points, faces, uvs, text


# convert textured mesh to point cloud
# treat vertices of mesh as points, get color from texture map, ignore displacement map
def sample_vertex_color(data_path, save_path):
    points, faces, uvs, text = read_mesh_obj(data_path)

    H, W, _ = text.shape
    uvs[:, 1] = 1 - uvs[:, 1]
    uvs = (uvs * np.array([[H, W]])).astype(int)

    colors = text[uvs[:, 1], uvs[:, 0], :]

    draw_mesh(points, colors, faces, visualize=False, save_path=save_path)
    return

# convert textured mesh to point cloud
# ignore displacement map (geometry details)
def sample_surface_color(data_path, save_path, num_points=1000000):
    mesh = trimesh.load_mesh(data_path)
    points, _, colors = trimesh.sample.sample_surface(mesh, num_points, sample_color=True)
    points = np.asarray(points)
    colors = np.asarray(colors)
    colors = colors[:, :-1] / 255.
    colors = np.clip(colors, a_min=0, a_max=1)
    draw_point_cloud(points, colors, visualize=False, save_path=save_path)
    return


# convert textured mesh to point cloud
# voxelize after random sampling
# ignore displacement map (geometry details)
def sample_voxel_color(data_path, save_path, num_points=1000000, num_voxel_axis=1000):
    mesh = trimesh.load_mesh(data_path)
    points, _, colors = trimesh.sample.sample_surface(mesh, num_points, sample_color=True)
    points = np.asarray(points)
    colors = np.asarray(colors)
    colors = colors[:, :-1] / 255.
    colors = np.clip(colors, a_min=0, a_max=1)
    points, colors = quantize_point_cloud(points, colors, num_voxel_axis)
    draw_point_cloud(points, colors, visualize=False, save_path=save_path)
    return


def generate_point_dataset(source_path, save_path, sampling="vox"):
    if sampling == "vert":
        mesh2point = sample_vertex_color
    elif sampling == "surf":
        mesh2point = sample_surface_color
    elif sampling == "vox":
        mesh2point = sample_voxel_color
    else:
        raise NotImplementedError

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_names = os.listdir(source_path)

    for file_name in file_names:
        print("Processing", file_name)

        file_path = source_path + file_name + '/models_reg/'
        save_file_path = save_path + file_name + '/'
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
        for exp_name in expressions:
            exp_path = file_path + exp_name + ".obj"
            save_exp_path = save_file_path + exp_name + ".ply"
            try:
                mesh2point(exp_path, save_exp_path)
            except Exception as e:  # ignore incomplete samples
                print("Error:", exp_path)
                sys.stdout.flush()
                pass
    return


def split_dataset():
    num_sample = 848
    num_train = int(0.8 * num_sample)
    num_valid = int(0.1 * num_sample)

    random.seed(0)
    publishable = [122, 212, 340, 344, 393, 395, 421, 527, 594, 610]
    pool = list(range(1, num_sample))
    test = []

    # publishable samples are used for testing
    for ele in publishable:
        pool.remove(ele)
        test.append(ele)

    random.shuffle(pool)
    train = pool[:num_train]
    valid = pool[num_train:num_train+num_valid]
    test.extend(pool[num_train+num_valid:])
    print("train", len(train), "valid", len(valid), "test", len(test))

    with open("FaceScape/train.txt", "w") as file:
        content = ''
        for ele in train:
            content += str(ele) + ","
        file.write(content[:-1])
    with open("FaceScape/valid.txt", "w") as file:
        content = ''
        for ele in valid:
            content += str(ele) + ","
        file.write(content[:-1])
    with open("FaceScape/test.txt", "w") as file:
        content = ''
        for ele in test:
            content += str(ele) + ","
        file.write(content[:-1])
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/work/lingdongwang_umass_edu/Datasets/FaceScape/")
    parser.add_argument('--sampling', type=str, default='vox')
    args = parser.parse_args()

    source_path = args.data_dir + "origin/TU-Model/"
    save_path =  args.data_dir + args.sampling + "/"

    # split_dataset()

    generate_point_dataset(source_path, save_path, sampling=args.sampling)