#!/bin/bash

if [ -n "$1" ]; then
  path="$1"
else
  path="/home/data"
fi

if [ -n "$2" ]; then
  size=$2
else
  size=64
fi

if [ -n "$3" ]; then
  epochs=$3
else
  epochs=20
fi

if [ -n "$4" ]; then
  warm=$4
else
  warm=2
fi

if [ -n "$5" ]; then
  recall="$5"
else
  recall="1,2,4,8"
fi

data_name=("car" "cub")
backbone_type=("resnet50" "inception" "googlenet")
feature_dim=(64 512)
ratios=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for data in ${data_name[*]}; do
  for backbone in ${backbone_type[*]}; do
    for feature in ${feature_dim[*]}; do
      for ratio in ${ratios[*]}; do
        echo "python train.py --data_path ${path} --data_name ${data} --backbone_type ${backbone} --feature_dim ${feature} --ratio ${ratio} --batch_size ${size} --num_epochs ${epochs} --warm_up ${warm} --recalls ${recall}"
        # shellcheck disable=SC2086
        python train.py --data_path ${path} --data_name ${data} --backbone_type ${backbone} --feature_dim ${feature} --ratio ${ratio} --batch_size ${size} --num_epochs ${epochs} --warm_up ${warm} --recalls ${recall}
      done
    done
  done
done
