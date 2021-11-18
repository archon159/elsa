Elsa: Energy-based Learning forSemi-supervised Anomaly Detection
================================================================
Official PyTorch implementation of "Elsa: Energy-based Learning forSemi-supervised Anomaly Detection" (BMVC 2021) by Sungwon Han*, Hyeonho Song*, Seungeon Lee, Sungwon Park, Meeyoung Cha.

Requirements
------------
### Environment
* python == 3.7
* torch == 1.7.1
* torchvision == 0.8.2
* CUDA == 10.1
* scikit-learn == 0.23.2
* tensorboard == 2.4.0
* torchlars == 0.1.2
* diffdist == 0.1
* soyclustering == 0.2.0

Pretraining
-----------
Please refer following repositories for original codes.

simCLR: https://github.com/google-research/simclr

CSI: https://github.com/alinlab/CSI

To pretrain simCLR or CSI on CIFAR-10, try the following commands.
```
cd elsa_finetune
```

### simCLR (ELSA)
```
python train.py --dataset cifar10 --model resnet18 \
--mode simclr --one_class_idx 0 --ratio_pollution 0.1 \
--batch_Size 512 --epochs 1000 --single_device 0

mv ./logs0/cifar10_resnet18_unsup_simclr_one_class_0 pretrained_result
```
  
### CSI (ELSA++)
```
python train.py --dataset cifar10 --model resnet18 \
--mode simclr_CSI --one_class_idx 0 --ratio_pollution 0.1 \
--batch_Size 128 --epochs 1000 --shift_trans_type rotation --single_device 0

mv ./logs0/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_0 pretrained_result
```

  
FineTuning
----------
To pretrain the model on CIFAR-10, try the following commands.
* ELSA uses pretrained model of simclr and ELSA++ uses pretrained model of simclr_CSI.
* Argument "known_normal" and "ratio_pollution" of finetuning must be same to "one_class_idx" and "ratio_pollution" in pretraining.


### ELSA
```
python ELSA.py --save_dir ./ --load_path ./pretrained_result/last.model \
--n_known_outlier 1 --known_normal 0 --known_outlier 1 \
--ratio_known_normal 0.1 --ratio_known_outlier 0.1 --ratio_pollution 0.1 \
 --batch_size 64 --n_cluster 50 --optimizer adam --lr 1e-4 --weight_decay 0.0 --n_epochs 50
```
  
### ELSA++
```
python ELSA.py --save_dir ./ --load_path ./pretrained_result/last.model \
--n_known_outlier 1 --known_normal 0 --known_outlier 1 \
--ratio_known_normal 0.1 --ratio_known_outlier 0.1 --ratio_pollution 0.1 \
 --batch_size 64 --n_cluster 50 --optimizer adam --lr 1e-4 --weight_decay 0.0 --n_epochs 50
```

License
-------
Distributed under the MIT License. See LICENSE.txt for more information.
