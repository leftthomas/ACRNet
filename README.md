# UnifiedProxy

A PyTorch implementation of Unified Proxy Loss based on SPL
paper [Unified Proxy Loss for Fine-grained Image Retrieval]().

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
```

- pretrainedmodels

```
pip install pretrainedmodels
```

- AdamP

```
pip install adamp
```

## Datasets

[CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
and [CUB200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
are used in this repo. You should download these datasets by yourself, and extract them into `${data_path}` directory,
make sure the dir names are `car` and `cub`. Then run `data_utils.py` to preprocess them.

## Usage

### Train Model

```
python train.py  --data_name cub --backbone_type inception --feature_dim 256
optional arguments:
--data_path                   datasets path [default value is '/home/data']
--data_name                   dataset name [default value is 'car'](choices=['car', 'cub'])
--backbone_type               backbone network type [default value is 'resnet50'](choices=['resnet50', 'inception', 'googlenet'])
--loss_name                   loss name [default value is 'balanced_proxy'](choices=['balanced_proxy', 'proxy_anchor'])
--feature_dim                 feature dim [default value is 512]
--batch_size                  training batch size [default value is 64]
--num_epochs                  training epoch number [default value is 20]
--warm_up                     warm up number [default value is 2]
--recalls                     selected recall [default value is '1,2,4,8']
```

You also could use `run.sh` to train all the combinations of hyper-parameters.

### Test Model

```
python test.py --retrieval_num 10
optional arguments:
--query_img_name              query image name [default value is '/home/data/car/uncropped/008055.jpg']
--data_base                   queried database [default value is 'car_resnet50_balanced_proxy_512_data_base.pth']
--retrieval_num               retrieval number [default value is 8]
```

## Benchmarks

The models are trained on one NVIDIA GeForce GTX 1070 (8G) GPU. `AdamP` is used to optimize the model, `lr` is `1e-2`
for the parameters of `proxies` and `1e-4` for other parameters, every `5 steps` the `lr` is reduced by `2`.
`scale` is `32`, a `layer_norm` op is injected to centering the embedding, other hyper-parameters are the default
values.

### CARS196

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@2</th>
      <th>R@4</th>
      <th>R@8</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">88.5%</td>
      <td align="center">93.1%</td>
      <td align="center">95.8%</td>
      <td align="center">97.7%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1ig6gwBBSm0EPzesL5KytYQ">5bww</a></td>
    </tr>
    <tr>
      <td align="center">Inception</td>
      <td align="center">85.5%</td>
      <td align="center">91.5%</td>
      <td align="center">95.0%</td>
      <td align="center">97.2%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1-wVIlNjiqiUUD1kRh8Efww">r6e7</a></td>
    </tr>
    <tr>
      <td align="center">GoogLeNet</td>
      <td align="center">78.1%</td>
      <td align="center">86.4%</td>
      <td align="center">91.5%</td>
      <td align="center">94.9%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1hMjWx9MG_40oHz6uBqe6OQ">espu</a></td>
    </tr>
  </tbody>
</table>

### CUB200

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@1</th>
      <th>R@2</th>
      <th>R@4</th>
      <th>R@8</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">ResNet50</td>
      <td align="center">67.7%</td>
      <td align="center">78.4%</td>
      <td align="center">85.8%</td>
      <td align="center">91.0%</td>
      <td align="center"><a href="https://pan.baidu.com/s/128SGDlxV1Cd8gPJEi7Z4gA">73h5</a></td>
    </tr>
    <tr>
      <td align="center">Inception</td>
      <td align="center">68.3%</td>
      <td align="center">78.7%</td>
      <td align="center">85.9%</td>
      <td align="center">90.8%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1i97a8vr3Le_9Bk-L0cTJug">u5b9</a></td>
    </tr>
    <tr>
      <td align="center">GoogLeNet</td>
      <td align="center">62.4%</td>
      <td align="center">73.0%</td>
      <td align="center">82.7%</td>
      <td align="center">89.3%</td>
      <td align="center"><a href="https://pan.baidu.com/s/1R6qnPfyBEKysCzWTdnO_6Q">anbq</a></td>
    </tr>
  </tbody>
</table>

## Results

![vis](results/result.png)
