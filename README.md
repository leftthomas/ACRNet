# RelationNet

A PyTorch implementation of RelationNet based on ICME 2022 paper
[Mining Relations for Weakly-Supervised Action Localization]().

![Network Architecture](result/structure.png)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch
```

- [MMAction2](https://mmaction2.readthedocs.io)

```
pip install git+https://github.com/open-mmlab/mim.git
mim install mmaction2
```

## Dataset

[THUMOS 14](http://crcv.ucf.edu/THUMOS14/download.html) and [ActivityNet](http://activity-net.org/download.html)
datasets are used in this repo, you should download these datasets from official websites. The RGB and Flow features of
these datasets are extracted by `utils.py` with `25FPS`. You should follow
[this link](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7) to install OpenCV4 with CUDA. And then
compile [denseFlow_GPU](https://github.com/daveboat/denseFlow_GPU), put the executable program in this dir. The options
could be found in `utils.py`, this script will take a lot of time to extract the features. Finally, I3D features of
these datasets are extracted by [this repo](https://github.com/Finspire13/pytorch-i3d-feature-extraction), the
`extract_features.py` file should be replaced with `extract.py`, the options could be found in `extract.py`. To make
this research friendly, we uploaded these I3D features
in [GoogleDrive](https://drive.google.com/drive/folders/1wudi03iJQYZ3qN2RUHB5senFrbFrWUQw?usp=sharing). You could
download them from there, and make sure the data directory structure is organized as follows:

 ```
├── thumos14                                    |  ├── activitynet
   ├── features                                  |   ├── features
       ├── val                                   |       ├── training 
           ├── video_validation_0000051_flow.npy |           ├── v___c8enCfzqw_flow.npy
           ├── video_validation_0000051_rgb.npy  |           ├── v___c8enCfzqw_rgb.npy
           └── ...                               |           └── ...                           
       ├── test                                  |       ├── validation                 
           ├── video_test_0000004_flow.npy       |           ├── v__1vYKA7mNLI_flow.npy  
           ├── video_test_0000004_rgb.npy        |           ├── v__1vYKA7mNLI_rgb.npy 
           └── ...                               |           └── ...     
   ├── videos                                    |   ├── videos  
       ├── val                                   |       ├── training      
           ├── video_validation_0000051.mp4      |           ├── v___c8enCfzqw.mp4
           └──...                                |           └──...        
       ├── test                                  |       ├── validation           
           ├── video_test_0000004.mp4            |           ├── v__1vYKA7mNLI.mp4
           └──...                                |           └──...      
   annotations.json                              | annotations_1.2.json, annotations_1.3.json
```

## Usage

You can easily train and test the model by running the script below. If you want to try other options, please refer to
`utils.py`.

### Train Model

```
python train.py --data_name activitynet1.2 --num_segments 50 --seed 0
```

### Test Model

```
python test.py --data_name thumos14 --model_file result/thumos14_model.pth
```

## Benchmarks

The models are trained on one NVIDIA GeForce GTX 1080 Ti GPU (11G). All the hyper-parameters are the default values.

### THUMOS14

<table>
<thead>
  <tr>
    <th rowspan="3">Method</th>
    <th colspan="8">THUMOS14</th>
    <th rowspan="3">Download</th>
  </tr>
  <tr>
    <td align="center">mAP@0.1</td>
    <td align="center">mAP@0.2</td>
    <td align="center">mAP@0.3</td>
    <td align="center">mAP@0.4</td>
    <td align="center">mAP@0.5</td>
    <td align="center">mAP@0.6</td>
    <td align="center">mAP@0.7</td>
    <td align="center">mAP@AVG</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">RelationNet</td>
    <td align="center">60.3</td>
    <td align="center">54.3</td>
    <td align="center">45.7</td>
    <td align="center">37.2</td>
    <td align="center">27.8</td>
    <td align="center">18.2</td>
    <td align="center">9.2</td>
    <td align="center">36.1</td>
    <td align="center"><a href="https://pan.baidu.com/s/1mv-RHb9VNu2FYBdzjNehPA">GoogleDrive</a></td>
  </tr>
</tbody>
</table>

mAP@AVG is the average mAP under the thresholds 0.1:0.1:0.7.

### ActivityNet

<table>
<thead>
  <tr>
    <th rowspan="3">Method</th>
    <th colspan="4">ActivityNet 1.2</th>
    <th colspan="4">ActivityNet 1.3</th>
    <th rowspan="3">Download</th>
  </tr>
  <tr>
    <td align="center">mAP@0.5</td>
    <td align="center">mAP@0.75</td>
    <td align="center">mAP@0.95</td>
    <td align="center">mAP@AVG</td>
    <td align="center">mAP@0.5</td>
    <td align="center">mAP@0.75</td>
    <td align="center">mAP@0.95</td>
    <td align="center">mAP@AVG</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">RelationNet</td>
    <td align="center">41.2</td>
    <td align="center">25.6</td>
    <td align="center">6.0</td>
    <td align="center">25.9</td>
    <td align="center">37.0</td>
    <td align="center">23.9</td>
    <td align="center">5.7</td>
    <td align="center">23.7</td>
    <td align="center"><a href="https://pan.baidu.com/s/11_7eu29IQ50rBU2W-dFceg">GoogleDrive</a></td>
  </tr>
</tbody>
</table>

mAP@AVG is the average mAP under the thresholds 0.5:0.05:0.95.
