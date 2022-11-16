## CU-Net

This is the official implementation of **CU-Net: Real-Time High-Fidelity Color Upsampling for Point Clouds**,
Lingdong Wang, Mohammad Hajiesmaili, Jacob Chakareski, Ramesh K. Sitaraman. 
The paper is available at https://arxiv.org/abs/2209.06112 .

## Prerequisite

Install prerequisites using the following command. 
Note that we are using PyTorch 1.11.0 + CUDA 11.3. 
Please adjust the version according to your environment.

```
pip install -r requirements.txt
```

## Datasets
### MPEG 8i
Download the MPEG 8i dataset from:
https://mpeg-pcc.org/index.php/pcc-content-database/8i-voxelized-surface-light-field-8ivslf-dataset/

### FaceScape

Apply for and download the FaceScape dataset from:
https://facescape.nju.edu.cn/

Organize the dataset as follows. Data splitting files (train.txt, valid.txt, test.txt) can be found in data/FaceScape/.


```
FaceScape\
    origin\
        TU-Model\
            1\
            2\
            ...
    train.txt
    valid.txt
    test.txt
```

Run the following command to generate a point cloud dataset from the orignal FaceScape dataset.

```
cd data/
python face_scape_preprocess.py --data_dir XXX/FaceScape/
```

## Test


Pretrained CU-Net models can be found in the pretrained/ folder.

Run the following command to test CU-Net and other baselines on the MPEG 8i dataset:

```
python mpeg8i.py --data_dir XXX/MPEG8i/
```

You can also test CU-Net on the FaceScape dataset by:

```
python test.py --data_dir XXX/FaceScape/
```

To evaluate baselines on the FaceScape dataset, use:

```
python baseline.py --data_dir XXX/FaceScape/
```

## Train


Use the following commands to reproduce the 2x, 5x, 10x upsampling tasks described in the paper.

```
python train.py --data_dir XXX/FaceScape/ --log_name face_vox_2x_b3c32 --block 3 --channel 32 --scale 2 --gt_scale 5 --batch_size 16
python train.py --data_dir XXX/FaceScape/ --log_name face_vox_5x_b4c64 --block 4 --channel 64 --scale 5 --gt_scale 2 --batch_size 8
python train.py --data_dir XXX/FaceScape/ --log_name face_vox_10x_b5c32 --block 5 --channel 32 --scale 10 --gt_scale 1 --batch_size 4
```

## Latency

Use the following command to measure the latency of CU-Net and devoxelization.

```
python latency_vox.py
```

Use the following command to measure the latency of other baseline methods.

```
python latency_traditional.py
```


