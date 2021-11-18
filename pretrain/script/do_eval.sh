#!/bin/bash

MODEL_DIR="/home/data/aya/elsa/pretrains_lars_simclr_csi"
MAIN_CMD="python3 ../eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score simclr --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix"
SAVE_DIR="./pretrain_eval"

CLASS_LIST=({0..9})
#RATIO_POLLUTION_LIST=('0.0' '0.05' '0.1')
RATIO_POLLUTION_LIST=('0.05')

usage() { echo "Usage: $0 [-e][-r]" 1>&2; exit 1; }
while getopts ":e:r" o; do
    case "${o}" in
        r)
            rm -rf ${SAVE_DIR}
            mkdir ${SAVE_DIR}
            exit
            ;;
        *)
            usage
            exit
            ;;
    esac
done
shift $((OPTIND-1))

for ratio_pollution in "${RATIO_POLLUTION_LIST[@]}"; do
    for one_class_idx in "${CLASS_LIST[@]}"; do
        CMD=${MAIN_CMD}" --one_class_idx ${one_class_idx} --load_path ${MODEL_DIR}/cifar10_resnet18_unsup_simclr_one_class_${one_class_idx}_ratio_pollution_${ratio_pollution}/last.model"
        echo ${CMD}
        ${CMD} | tee ${SAVE_DIR}/simclr_lars_one_class_${one_class_idx}_ratio_pollution_${ratio_pollution}
    done
done
