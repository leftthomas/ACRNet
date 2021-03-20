# OSSCo

A PyTorch implementation of OSSCo based on ICCV 2021 paper [Fully Unsupervised Domain-Agnostic Image Retrieval]().

![Network Architecture](result/structure.jpg)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch
```

- [Pytorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/)

```
pip install pytorch-metric-learning
```

- [THOP](https://github.com/Lyken17/pytorch-OpCounter)

```
pip install thop
```

## Dataset

[Cityscapes FoggyDBF](https://www.cityscapes-dataset.com/) and
[CUFSF](http://mmlab.ie.cuhk.edu.hk/archive/cufsf/) datasets are used in this repo, you could download these datasets
from official websites, or download them from [MEGA](https://mega.nz/folder/5sQD0QqK#zM5GfOSPvpPpfquGJd8Vjg). The data
should be rearranged, please refer the paper to acquire the details of `train/val` split. The data directory structure
is shown as follows:

 ```
 cityscapes
    ├── train
       ├── clear (clear images)
           ├── aachen_000000_000019_leftImg8bit.png
           └── ...
           ├── bochum_000000_000313_leftImg8bit.png
           └── ...
       ├── fog (fog images)
           same structure as clear
           ...         
    ├── val
       same structure as train
   ...
cufsf
    same structure as cityscapes
```

## Usage

```
python main.py or comp.py --data_name cufsf
optional arguments:
# common args
--data_root                   Datasets root path [default value is 'data']
--data_name                   Dataset name [default value is 'cityscapes'](choices='cityscapes', 'cufsf'])
--method_name                 Compared method name [default value is 'ossco'](choices=['ossco', 'simclr', 'npid', 'proxyanchor', 'softtriple', 'pretrained'])
--proj_dim                    Projected feature dim for computing loss [default value is 128]
--temperature                 Temperature used in softmax [default value is 0.1]
--batch_size                  Number of images in each mini-batch [default value is 16]
--total_iter                  Number of bp to train [default value is 10000]
--ranks                       Selected recall to val [default value is [1, 2, 4, 8]]
--save_root                   Result saved root path [default value is 'result']
# args for ossco
--style_num                   Number of used styles [default value is 8]
--gan_iter                    Number of bp to train gan model [default value is 4000]
--rounds                      Number of round to train whole model [default value is 5]
```

For example, to train `npid` on `cufsf` dataset, report `R@1` and `R@5`:

```
python comp.py --method_name npid --data_name cufsf --batch_size 64 --ranks 1 5
```

to train `ossco` on `cityscapes` dataset, with `16` random selected styles:

```
python main.py --method_name ossco --data_name cityscapes --style_num 16
```

## Benchmarks

The models are trained on one NVIDIA GTX TITAN (12G) GPU. `Adam` is used to optimize the model, `lr` is `1e-3`
and `weight decay` is `1e-6`. `batch size` is `16` for `ossco`, `32` for `simclr`, `64` for `npid`.
`lr` is `2e-4` and `betas` is `(0.5, 0.999)` for GAN, other hyper-parameters are the default values.

### Cityscapes

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="4">Clear --&gt; Foggy</th>
    <th colspan="4">Foggy --&gt; Clear</th>
    <th colspan="4">Clear &lt;--&gt; Foggy</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Pretrained</td>
    <td align="center">77.0</td>
    <td align="center">82.0</td>
    <td align="center">86.6</td>
    <td align="center">89.2</td>
    <td align="center">93.4</td>
    <td align="center">95.6</td>
    <td align="center">97.0</td>
    <td align="center">98.4</td>
    <td align="center">45.7</td>
    <td align="center">53.3</td>
    <td align="center">59.3</td>
    <td align="center">65.4</td>
    <td align="center"><a href="https://pan.baidu.com/s/1G9qdUvrFHqEm1kbmPmel9w">ea3u</a></td>
  </tr>
  <tr>
    <td align="center">NPID</td>
    <td align="center">22.8</td>
    <td align="center">29.4</td>
    <td align="center">37.2</td>
    <td align="center">46.4</td>
    <td align="center">21.6</td>
    <td align="center">28.4</td>
    <td align="center">35.6</td>
    <td align="center">43.6</td>
    <td align="center">5.9</td>
    <td align="center">8.3</td>
    <td align="center">11.2</td>
    <td align="center">14.1</td>
    <td align="center"><a href="https://pan.baidu.com/s/1PWLOBKWb8gUUibXOX9OQyA">hu2k</a></td>
  </tr>
  <tr>
    <td align="center">SimCLR</td>
    <td align="center">92.2</td>
    <td align="center">94.6</td>
    <td align="center">96.6</td>
    <td align="center">97.8</td>
    <td align="center">89.6</td>
    <td align="center">93.0</td>
    <td align="center">95.4</td>
    <td align="center">98.2</td>
    <td align="center">80.1</td>
    <td align="center">85.4</td>
    <td align="center">88.8</td>
    <td align="center">92.3</td>
    <td align="center"><a href="https://pan.baidu.com/s/1aJGLPODKE4cCHLZYDg96jA">4jvm</a></td>
  </tr>
  <tr>
    <td align="center">SoftTriple</td>
    <td align="center">99.6</td>
    <td align="center">99.8</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">99.8</td>
    <td align="center">99.8</td>
    <td align="center">99.8</td>
    <td align="center">100</td>
    <td align="center">98.4</td>
    <td align="center">99.7</td>
    <td align="center">99.8</td>
    <td align="center">99.9</td>
    <td align="center"><a href="https://pan.baidu.com/s/1mYIRpX4ABX9YVLs0gFJVmg">6we5</a></td>
  </tr>
  <tr>
    <td align="center">ProxyAnchor</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center"><a href="https://pan.baidu.com/s/1aEQhoDH3ciAHESbzSfeR6Q">99k3</a></td>
  </tr>
  <tr>
    <td align="center">OSSCo</td>
    <td align="center"><b>98.6</b></td>
    <td align="center"><b>99.2</b></td>
    <td align="center"><b>99.6</b></td>
    <td align="center"><b>99.8</b></td>
    <td align="center"><b>99.0</b></td>
    <td align="center"><b>99.4</b></td>
    <td align="center"><b>99.6</b></td>
    <td align="center"><b>99.6</b></td>
    <td align="center"><b>97.0</b></td>
    <td align="center"><b>98.6</b></td>
    <td align="center"><b>99.2</b></td>
    <td align="center"><b>99.5</b></td>
    <td align="center"><a href="https://pan.baidu.com/s/19d3v1PTnX-Z3dH7ifeY1oA">cb2b</a></td>
  </tr>
</tbody>
</table>

### CUFSF

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="4">Sketch --&gt; Image</th>
    <th colspan="4">Image --&gt; Sketch</th>
    <th colspan="4">Sketch &lt;--&gt; Image</th>
    <th rowspan="2">Download</th>
  </tr>
  <tr>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
    <td align="center">R@1</td>
    <td align="center">R@2</td>
    <td align="center">R@4</td>
    <td align="center">R@8</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Pretrained</td>
    <td align="center">9.0</td>
    <td align="center">13.1</td>
    <td align="center">18.1</td>
    <td align="center">25.6</td>
    <td align="center">16.6</td>
    <td align="center">24.1</td>
    <td align="center">30.7</td>
    <td align="center">38.2</td>
    <td align="center">0.3</td>
    <td align="center">0.3</td>
    <td align="center">1.3</td>
    <td align="center">3.0</td>
    <td align="center"><a href="https://pan.baidu.com/s/1td9R3zV1buYg5ekvaEaMSg">imi4</a></td>
  </tr>
  <tr>
    <td align="center">NPID</td>
    <td align="center">37.2</td>
    <td align="center">48.7</td>
    <td align="center">63.8</td>
    <td align="center">73.4</td>
    <td align="center">40.7</td>
    <td align="center">52.8</td>
    <td align="center">67.8</td>
    <td align="center">73.9</td>
    <td align="center">27.1</td>
    <td align="center">34.4</td>
    <td align="center">46.0</td>
    <td align="center">60.1</td>
    <td align="center"><a href="https://pan.baidu.com/s/1MKLAWG4x-tr-9T7M2exUFg">xvci</a></td>
  </tr>
  <tr>
    <td align="center">SimCLR</td>
    <td align="center">24.1</td>
    <td align="center">39.2</td>
    <td align="center">56.3</td>
    <td align="center">72.4</td>
    <td align="center">32.7</td>
    <td align="center">45.2</td>
    <td align="center">56.3</td>
    <td align="center">68.8</td>
    <td align="center">15.1</td>
    <td align="center">21.9</td>
    <td align="center">33.7</td>
    <td align="center">49.0</td>
    <td align="center"><a href="https://pan.baidu.com/s/1WzYf-QmAB1YjfEMLdkAIeg">xtux</a></td>
  </tr>
  <tr>
    <td align="center">SoftTriple</td>
    <td align="center">86.4</td>
    <td align="center">92.5</td>
    <td align="center">95.5</td>
    <td align="center">99.0</td>
    <td align="center">89.4</td>
    <td align="center">93.5</td>
    <td align="center">97.5</td>
    <td align="center">99.5</td>
    <td align="center">79.6</td>
    <td align="center">85.9</td>
    <td align="center">92.7</td>
    <td align="center">96.2</td>
    <td align="center"><a href="https://pan.baidu.com/s/1L7iUrQmtzlaSOVqLjfv-Tw">5qb9</a></td>
  </tr>
  <tr>
    <td align="center">ProxyAnchor</td>
    <td align="center">95.5</td>
    <td align="center">98.0</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">95.5</td>
    <td align="center">98.5</td>
    <td align="center">100</td>
    <td align="center">100</td>
    <td align="center">91.7</td>
    <td align="center">95.7</td>
    <td align="center">98.2</td>
    <td align="center">99.7</td>
    <td align="center"><a href="https://pan.baidu.com/s/1YWRsng6X9lq1yVNbJv6aVQ">inai</a></td>
  </tr>
  <tr>
    <td align="center">OSSCo</td>
    <td align="center"><b>82.4</b></td>
    <td align="center"><b>93.5</b></td>
    <td align="center"><b>97.5</b></td>
    <td align="center"><b>99.5</b></td>
    <td align="center"><b>88.4</b></td>
    <td align="center"><b>98.0</b></td>
    <td align="center"><b>99.5</b></td>
    <td align="center"><b>99.5</b></td>
    <td align="center"><b>55.0</b></td>
    <td align="center"><b>70.4</b></td>
    <td align="center"><b>87.2</b></td>
    <td align="center"><b>94.5</b></td>
    <td align="center"><a href="https://pan.baidu.com/s/1Jh0zTifYl2ul9__R7WrSuw">q6ji</a></td>
  </tr>
</tbody>
</table>

### T-SNE (CUFSF)

![tsne](result/tsne.png)
