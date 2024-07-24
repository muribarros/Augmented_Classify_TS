#!/bin/bash

__conda_setup="$('/home/kaue.duarte/software/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/kaue.duarte/software/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/kaue.duarte/software/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/kaue.duarte/software/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

((array_id=${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}))
conda activate py_310

((i=SLURM_ARRAY_TASK_ID-1))

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

image_type=$1  # Whole_Brain, DKT, Slant
network_name=$2 #VGG16, VGG19, ResNet40
loss=$3 #binary_crossentropy
metric=$4 #accuracy
transferlearning=$5 #None, ImageNet, RadImageNet
python run_monai_wandb.py $image_type $network_name $loss $metric $transferlearning
#python run_monai.py $image_type $network_name $loss $metric $transferlearning

