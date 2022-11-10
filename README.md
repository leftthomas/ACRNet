# ACRNet

A PyTorch implementation of ACRNet based on ICME 2023 paper
[Weakly-supervised Temporal Action Localization with Adaptive Clustering and Refining Network]().

![Network Architecture](result/model.png)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

- [MMAction2](https://mmaction2.readthedocs.io)

```
pip install openmim
mim install mmaction2 -f https://github.com/open-mmlab/mmaction2.git
```

## Dataset

[THUMOS 14](http://crcv.ucf.edu/THUMOS14/download.html) and [ActivityNet](http://activity-net.org/download.html)
datasets are used in this repo, you should download these datasets from official websites. The RGB and Flow features of
these datasets are extracted by `dataset.py` with `25 FPS`. You should follow
[this link](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7) to install OpenCV4 with CUDA. And then
compile [denseFlow_GPU](https://github.com/daveboat/denseFlow_GPU), put the executable program in this dir. The options
could be found in [dataset.py](dataset.py), this script will take a lot of time to extract the features. Finally, I3D
features of these datasets are extracted by [this repo](https://github.com/Finspire13/pytorch-i3d-feature-extraction),
the `extract_features.py` file should be replaced with [extract.py](extract.py), the options could be found in
[extract.py](extract.py). To make this research friendly, we uploaded these I3D features in
[MEGA](https://mega.nz/folder/6sFxjaZB#Jtx69Kb2RHu2ldXoNzsODQ). You could download them from there, and make sure the
data directory structure is organized as follows:

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
   annotations.json                              |    annotations_1.2.json, annotations_1.3.json
```

## Usage

You can easily train and test the model by running the script below. If you want to try other options, please refer to
[utils.py](utils.py).

### Train Model

```
python main.py --max-seqlen 500 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --path-dataset path/to/Thumos14 --num-class 20 --use-model CO2  --max-iter 5000 --weight_decay 0.001 --model-name CO2_3552 --seed 3552
python main.py --k 5  --dataset-name ActivityNet1.2 --path-dataset path/to/ActivityNet1.2 --num-class 100 --use-model ANT_CO2 --lr 3e-5 --max-seqlen 60 --model-name ANT_CO2_3552 --seed 3552 --max-iter 22000
```

### Test Model

```
python test.py --dataset-name Thumos14reduced --num-class 20  --path-dataset path/to/Thumos14  --use-model CO2 --model-name CO2_3552
python test.py --dataset-name ActivityNet1.2 --num-class 100 --path-dataset path/to/ActivityNet1.2 --use-model ANT_CO2 --model-name ANT_CO2_3552 --max-seqlen 60
```

## Benchmarks

The models are trained on one NVIDIA GeForce RTX 3090 GPU (24G). All the hyper-parameters are the default values.

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
    <td align="center"><a href="https://pan.baidu.com/s/1mv-RHb9VNu2FYBdzjNehPA">MEGA</a></td>
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
    <td align="center"><a href="https://pan.baidu.com/s/11_7eu29IQ50rBU2W-dFceg">MEGA</a></td>
  </tr>
</tbody>
</table>

mAP@AVG is the average mAP under the thresholds 0.5:0.05:0.95.
