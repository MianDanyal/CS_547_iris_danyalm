# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# data folder
DATASET_ROOT='datasets/fipt/indoor_synthetic/'
DATASET='synthetic'
# scene name
SCENE='bathroom'
LDR_IMG_DIR='Image'
EXP='fipt_syn_bathroom_huber'
VAL_FRAME=10 
CRF_BASIS=3
# whether has part segmentation
HAS_PART=1
SPP=32
spp=16
RES_SCALE=0.25

python render.py --experiment_name $EXP --device 0\
        --ckpt last_1.ckpt \
        --dataset $DATASET $DATASET_ROOT$SCENE \
        --emitter_path checkpoints/$EXP/bake\
        --output_path 'outputs/'$EXP'/output'\
        --split 'val'\
        --ldr_img_dir $LDR_IMG_DIR \
        --res_scale $RES_SCALE --num_workers 2 \
        --SPP $SPP --spp $spp --crf_basis $CRF_BASIS 
