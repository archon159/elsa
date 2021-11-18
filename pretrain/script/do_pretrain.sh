#!/bin/bash

#MODE='simclr_CSI'
MODE='simclr'
SHIFT_TRANS_TYPE='rotation'
CLASS_LIST=({0..9})
RATIO_POLLUTION_LIST=('0.0' '0.05' '0.1')
EPOCHS=1000
TRAIN_STR="../train.py --dataset cifar10 --model resnet18"
RESULT_STR="cifar10_resnet18_unsup"

usage() { echo "Usage: $0 [-g <GPU to use>][-r]" 1>&2; exit 1; }
while getopts ":g:r" o; do
    case "${o}" in
        g)
            GPU_NUM=${OPTARG}
            ;;
        r)
            rm -rf ./logs*
            rm -rf pretrain_result
            mkdir pretrain_result
            exit
            ;;
        *)
            usage
            exit
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${GPU_NUM}" ]; then
    usage
    exit
fi

idx=0
for known_normal in "${CLASS_LIST[@]}"; do
    for ratio_pollution in "${RATIO_POLLUTION_LIST[@]}"; do
        let TARGET_GPU=${idx}%4
        if [ $GPU_NUM -eq $TARGET_GPU ]; then
            if [ ${MODE} = 'simclr_CSI' ]; then
                echo "python3 ${TRAIN_STR} --batch_size 128 --mode ${MODE} --single_device $GPU_NUM --one_class_idx $known_normal --ratio_pollution $ratio_pollution --epochs ${EPOCHS} --shift_trans_type ${SHIFT_TRANS_TYPE}"
                python3 ${TRAIN_STR} --batch_size 128 --mode ${MODE} --single_device $GPU_NUM --one_class_idx $known_normal --ratio_pollution $ratio_pollution --epochs ${EPOCHS} --shift_trans_type ${SHIFT_TRANS_TYPE}
                mv ./logs${GPU_NUM}/${RESULT_STR}_${MODE}_shift_${SHIFT_TRANS_TYPE}_one_class_${known_normal} ./pretrain_result/${RESULT_STR}_${MODE}_shift_${SHIFT_TRANS_TYPE}_one_class_${known_normal}_ratio_pollution_${ratio_pollution}

            elif [ ${MODE} = 'simclr' ]; then
                echo "python3 ${TRAIN_STR} --batch_size 512 --mode ${MODE} --single_device $GPU_NUM --one_class_idx $known_normal --ratio_pollution $ratio_pollution --epochs ${EPOCHS}"
                 python3 ${TRAIN_STR} --batch_size 512 --mode ${MODE} --single_device $GPU_NUM --one_class_idx $known_normal --ratio_pollution $ratio_pollution --epochs ${EPOCHS}
                mv ./logs${GPU_NUM}/${RESULT_STR}_${MODE}_one_class_${known_normal} ./pretrain_result/${RESULT_STR}_${MODE}_one_class_${known_normal}_ratio_pollution_${ratio_pollution}
            else
                echo 'Wrong argument for MODE'
                exit
            fi
        fi
        let idx=${idx}+1
    done
done

