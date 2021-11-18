#!/bin/bash

True=1
False=0

#python3 ELSApp_earlystop.py --save_dir ELSApp_final/esnew_0_1 --n_cluster 500 --load_path /home/data/aya/elsa/pretrains_lars_csi/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_0_ratio_pollution_0.0/last.model --optimizer adam --lr 1e-4 --n_epochs 50 | tee esnew.txt

TRAIN_MODE='ELSApp'
SAVE_DIR='./ELSApp_result'
#LOAD_DIR='/home/data/aya/elsa/pretrains_lars_simclr_csi'
LOAD_DIR='/home/data/aya/elsa/pretrains_lars_csi'
N_CLUSTER=50
OPTIMIZER='adam'
LEARNING_RATE='1e-4'
WEIGHT_DECAY='0.0'
BATCH_SIZE=64
N_EPOCHS=50

CLASS_LIST=({0..9})

RATIO_KNOWN_LIST=('0.01' '0.05' '0.1')
RATIO_POLLUTION_LIST=('0.0' '0.05' '0.1')

usage() { echo "Usage: $0 [-n <Number of Servers>][-i <Server Index>][-r]" 1>&2; exit 1; }
while getopts ":n:i:r" o; do
    case "${o}" in
        n)
            SERVER_NUM=${OPTARG}
            ;;
        i)
            SERVER_IDX=${OPTARG}
            ;;
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

if [ -z "${SERVER_NUM}" ]; then
    usage
    exit
fi

if [ -z "${SERVER_IDX}" ]; then
    usage
    exit
fi

idx=0
for known_normal in "${CLASS_LIST[@]}"; do
    let TARGET_SERVER=${idx}%${SERVER_NUM}
    let idx=${idx}+1
    if [ $SERVER_IDX -ne $TARGET_SERVER ]; then
        continue
    fi

    for known_outlier in "${CLASS_LIST[@]}"; do
        if [ ${known_normal} -eq ${known_outlier} ]; then
            continue
        fi

        for ratio_known in "${RATIO_KNOWN_LIST[@]}"; do
            for ratio_pollution in "${RATIO_POLLUTION_LIST[@]}"; do
                SAVE_PATH="${SAVE_DIR}/${TRAIN_MODE}_known_normal_${known_normal}_known_outlier_${known_outlier}_ratio_known_normal_${ratio_known}_ratio_known_outlier_${ratio_known}_ratio_pollution_${ratio_pollution}"
                mkdir -p ${SAVE_PATH}
                if [ ${TRAIN_MODE} = 'ELSApp' ]; then
                    LOAD_PREFIX="cifar10_resnet18_unsup_simclr_CSI_shift_rotation"
                elif [ ${TRAIN_MODE} = 'ELSA' ]; then
                    LOAD_PREFIX="cifar10_resnet18_unsup_simclr"
                else
                    echo 'Wrong argument for TRAIN_MODE'
                    exit
                fi
                LOAD_PATH="${LOAD_DIR}/${LOAD_PREFIX}_one_class_${known_normal}_ratio_pollution_${ratio_pollution}/last.model"
                CMD="python ../${TRAIN_MODE}.py --save_dir ${SAVE_PATH} --batch_size ${BATCH_SIZE} --n_cluster ${N_CLUSTER} --load_path ${LOAD_PATH} --n_known_outlier 1 --known_normal ${known_normal} --known_outlier ${known_outlier} --ratio_known_normal ${ratio_known} --ratio_known_outlier ${ratio_known} --ratio_pollution ${ratio_pollution} --optimizer ${OPTIMIZER} --lr ${LEARNING_RATE} --weight_decay ${WEIGHT_DECAY} --n_epochs ${N_EPOCHS}"
                echo ${CMD}

                CUDA_VISIBLE_DEVICES=0,1,2,3 ${CMD} | tee ${SAVE_PATH}/log
            done
        done
    done
done
