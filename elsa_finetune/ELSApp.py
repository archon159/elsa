import sys, os
import pandas as pd
import utils,json
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
import torch.optim as optim, copy

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import model_csi as C
from dataloader_es import *
from parser import * 
import transform_layers as TL
import torch.optim.lr_scheduler as lr_scheduler
from soyclustering import SphericalKMeans
from scipy import sparse
from randaugment_without_rotation import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)
import random,numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

### helper functions
def checkpoint(f,  tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "protos": f.module.prototypes
    }
    torch.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)

def energy_score(z, model):
    zp = model.module.prototypes
    logits = torch.matmul(z, zp.t()) / args.temperature
    Le = torch.log(torch.exp(logits).sum(dim=1))
    return Le

def cal_class_auroc(nd1,nd2,nd3,nd4,nd5,and1,and2,and3,and4,and5,ndsum,andsum,ndmul,andmul,cls_list):
    # Class AUROC
    normal_class = args.known_normal
    anomaly_classes = [i for i in range(args.n_classes)]
    anomaly_classes.remove(normal_class)
    tod1_average = 0
    for anomaly in anomaly_classes:
        tod1 = nd1 + np.array(and1)[np.array(cls_list) == anomaly].tolist()
        total_label = [1 for i in range(len(nd1))] + [0 for i in range(len(tod1) - len(nd1))]
        print('---------------------- Evaluation class: {} --------------------------'.format(anomaly))
        print(len(nd1), len(tod1) - len(nd1))
        print("px\t", roc_auc_score(total_label, tod1))
        print('----------------------------------------------------------------------')
        print()
        tod1_average  += roc_auc_score(total_label, tod1)
    tod1_average /= len(anomaly_classes)

    print('------------------- Evaluation class average --------------------')
    print(len(nd1), len(tod1) - len(nd1))
    print("px\t", tod1_average)
    print('----------------------------------------------------------------------')
    print()
    return 
def get_features(pos_1,model,use_simclr_aug=False,use_ensemble=False):
    if use_ensemble:
        sample_num = args.sample_num
    out_ensemble, pen_out_ensemble = [], []
    for seed in range(args.sample_num): # for ensemble run  N times
        set_random_seed(seed) # random seed setting
        images1 = torch.cat([rotation(hflip(pos_1), k) for k in range(4)]) # 4B
        if use_simclr_aug:
            images1 = simclr_aug(images1) 
        _, outputs_aux = model(images1, simclr=True, penultimate=True, shift=True)
        out = outputs_aux['simclr'] # 4B, D
        pen_out = outputs_aux['shift'] # 4B, D
                
        out_ensemble.append(out) 
                
        pen_out_ensemble.append(pen_out)
            ## ensembling 
    out = torch.stack(out_ensemble,dim=1).mean(dim=1) # N D
    pen_out = torch.stack(pen_out_ensemble,dim=1).mean(dim=1) # N, 4
    norm_out = F.normalize(out,dim=-1)

    return out,pen_out,norm_out

def generate_prototypes(model, valid_loader, n_cluster=100, split = False):
    first = True
    with torch.no_grad():
        for idx, (pos_1, _, _, semi_target,_, _) in enumerate(valid_loader):
            pos_1 = pos_1.cuda(non_blocking=True) # B
            images1 = torch.cat([rotation(pos_1, k) for k in range(4)]) # 4B

            _, outputs_aux = model(images1, simclr=True, penultimate=False, shift=True)
            out = F.normalize(outputs_aux['simclr'],dim=-1)

            all_semi_target = semi_target.repeat(4)
            true_out = out[all_semi_target != -1,:]    
            true_out_list = torch.stack(true_out.chunk(4, dim = 0), dim = 1) # [B*D, B*D, B*D, B*D] -> B*4*D

            if first:
                all_out_list = true_out_list
                first = False
            else:
                all_out_list = torch.cat((all_out_list, true_out_list), dim = 0)
    # Set prototypes (k-means++)
    all_out_numpy = all_out_list.cpu().numpy() # T * 4 * D
    proto_list = []
    all_out = all_out_numpy.reshape(-1, all_out_numpy.shape[2])
    all_out_sp = sparse.csr_matrix(all_out)
    print(sum(np.isnan(all_out)))

    while True:
        try:
            spherical_kmeans = SphericalKMeans(
                n_clusters=n_cluster,
                max_iter=10,
                verbose=1,
                init='similar_cut'
            )

            spherical_kmeans.fit(all_out_sp)
            break
        except KeyboardInterrupt:
            assert 0
        except:
            print("K-means failure... Retrying")
            continue    
    protos = spherical_kmeans.cluster_centers_
    protos = F.normalize(torch.Tensor(protos), dim = -1)
    return protos.to(device)

def earlystop_score(model,validation_dataset):
    rot_num = 4
    weighted_aucscores,aucscores = [],[]
    zp = model.module.prototypes
    for images1,images2, semi_target in validation_dataset:
        prob,prob2, label_list = [] , [], []
        weighted_prob, weighted_prob2 = [], []
        Px_mean,Px_mean2 = 0, 0  
        all_semi_targets = torch.cat([semi_target,semi_target+1])

        _, outputs_aux = model(images1, simclr=True, penultimate=False, shift=True)
        norm_out = F.normalize(outputs_aux['simclr'],dim=-1)

        logits = torch.matmul(norm_out, zp.t()) # (B + B + B + B, # of P)
        logits_list = logits.chunk(rot_num, dim = 0) # list of (B, # of P)
        out_list = norm_out.chunk(rot_num, dim = 0)

        for shi in range(rot_num):
            # Energy / Similar to P(x)
            Px_mean += torch.log(torch.exp(logits_list[shi]).sum(dim=1)) 
        prob.extend(Px_mean.tolist())
        _, outputs_aux = model(images2, simclr=True, penultimate=False, shift=True)
        norm_out = F.normalize(outputs_aux['simclr'],dim=-1)
        logits = torch.matmul(norm_out, zp.t()) # (B + B + B + B, # of P)
        logits_list = logits.chunk(rot_num, dim = 0) # list of (B, # of P)
        out_list = norm_out.chunk(rot_num, dim = 0) 

        for shi in range(rot_num):
            # Energy / Similar to P(x)
            Px_mean2 += torch.log(torch.exp(logits_list[shi]).sum(dim=1)) 
        prob2.extend(Px_mean2.tolist())

        label_list.extend(all_semi_targets)
        aucscores.append(roc_auc_score(label_list, prob2+prob))
    print("earlystop_score:",np.mean(aucscores))

    return np.mean(aucscores)

def test(model, test_loader, train_loader, epoch):
    model.eval()
    with torch.no_grad():
        ndsum, ndmul, nd1, nd2, nd3, nd4,nd5 = [], [], [], [], [], [], []
        andsum, andmul, and1, and2, and3, and4,and5 = [], [], [], [], [], [], []
        cls_list = []

        for idx, (pos_1, _, target,  _,cls,_) in enumerate(test_loader):
            
            negative_target = (target == 1).nonzero().squeeze()
            positive_target = (target != 1).nonzero().squeeze()
            
            pos_1 = pos_1.cuda(non_blocking=True) # B
            zp = model.module.prototypes
            
            out, pen_out, norm_out = get_features(pos_1,model,use_simclr_aug=True,use_ensemble=True)

            logits = torch.matmul(norm_out, zp.t()) # (B + B + B + B, # of P)
            logits_list = logits.chunk(4, dim = 0) # list of (B, # of P)
            out_list = norm_out.chunk(4, dim = 0) 
            pen_out_list = pen_out.chunk(4, dim = 0) # (B, 4)의 list
            Px_mean = 0
            for shi in range(4):
                # Energy / Similar to P(x)
                Px_mean += torch.log(torch.exp(logits_list[shi]).sum(dim=1))  #* all_weight_energy[shi]

            cls_list.extend(cls[negative_target])
            if len(positive_target.shape) != 0:
                nd1.extend(Px_mean[positive_target].tolist())
            if len(negative_target.shape) != 0:
                and1.extend(Px_mean[negative_target].tolist())

    cal_class_auroc(nd1,nd2,nd3,nd4,nd5,and1,and2,and3,and4,and5,ndsum,andsum,ndmul,andmul,cls_list)
    return
    
## 0) setting 
seed = args.seed
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
utils.makedirs(args.save_dir)
with open(f'{args.save_dir}/params.txt', 'w') as f: # training setting saving
    json.dump(args.__dict__, f)
if args.print_to_log: # 
    sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

args.device = device
    
## 1) pretraining & prototyping
args.shift_trans, args.K_shift = C.get_shift_module()
args.shift_trans = args.shift_trans.to(device)

if args.dataset == 'cifar10':
    args.image_size = (32, 32, 3)
else:
    raise

model = C.get_classifier('resnet18', n_classes=10).to(device)
model = C.get_shift_classifer(model, 4).to(device)
simclr_aug = C.get_simclr_augmentation(args, image_size=args.image_size).to(device)

rotation = args.shift_trans 
criterion = nn.CrossEntropyLoss()


    
if args.load_path != None: # pretrained model loading
    ckpt_dict = torch.load(args.load_path)
    model.load_state_dict(ckpt_dict, strict = True)
else:
    assert False , "Not implemented error: you should give pretrained and prototyped model"
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(args.device)

# Transformation step followed by CSI (Rotation -> Augmentation -> Normalization)
train_transform = transforms.Compose([
    transforms.Resize((args.image_size[0], args.image_size[1])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((args.image_size[0], args.image_size[1])),
    transforms.ToTensor(),
])


# dataset loader
strong_aug = RandAugmentMC(n=12,m=5)
total_dataset = load_dataset("~/data", normal_class=[args.known_normal], known_outlier_class=args.known_outlier,
                             n_known_outlier_classes=args.n_known_outlier, ratio_known_normal=args.ratio_known_normal,
                             ratio_known_outlier=args.ratio_known_outlier, ratio_pollution=args.ratio_pollution, random_state=None,
                             train_transform=train_transform, test_transform=test_transform,
                            valid_transform=strong_aug)


train_loader, false_valid_loader ,valid_loader, test_loader = total_dataset.loaders(batch_size = args.batch_size)

# Set prototypes (naive)
print("setup fixed validation data")
rot_num = 4
validation_dataset = []
for i, (pos,pos2,_, semi_target,_,_) in tqdm(enumerate(valid_loader)):
    images1 = torch.cat([rotation(pos, k) for k in range(rot_num)])
    images2 = torch.cat([rotation(pos2, k) for k in range(rot_num)])
    images1 = images1.to(device)
    images2 = images2.to(device)
    images1 = simclr_aug(images1)
    images2 = simclr_aug(images2)
    val_semi_target = torch.zeros(len(semi_target), dtype=torch.int64)
    validation_dataset.append([images1,images2,val_semi_target])

print("kmeans ",args.n_cluster)
n_cluster = args.n_cluster
protos = generate_prototypes(model, false_valid_loader, n_cluster=n_cluster, split=False)
model.module.prototypes = protos
model.module.prototypes = model.module.prototypes.to(args.device)

params = model.parameters()
if args.optimizer == 'sgd':
    optim = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_decay_gamma = 0.1
elif args.optimizer == 'adam':
    optim = optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .999), weight_decay=args.weight_decay)
    lr_decay_gamma = 0.3
elif args.optimizer == 'lars':
    from torchlars import LARS
    base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optim = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    lr_decay_gamma = 0.1
elif args.optimizer == 'ranger':
    from ranger import Ranger
    optim = Ranger(model.parameters(), weight_decay=args.weight_decay,lr=args.lr)
else:
    raise NotImplementedError()
    
print("known_normal:",args.known_normal,"known_outlier:",args.known_outlier)
rotation = args.shift_trans 
criterion = nn.CrossEntropyLoss()

earlystop_trace = []
end_train = False
max_earlystop_auroc = 0
for epoch in range(args.n_epochs):
    model.train()  
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    # training
    losses_energy = []
    losses_shift = []
    for i, (pos, _, _, semi_target, _,_) in tqdm(enumerate(train_loader)):
        pos = pos.to(device)
        semi_target = semi_target.to(device)
        batch_size = pos.size(0)
        pos_1, pos_2 = hflip(pos.repeat(2, 1, 1, 1)).chunk(2)  # hflip              
        
        images1 = torch.cat([rotation(pos_1, k) for k in range(4)])
        images2 = torch.cat([rotation(pos_2, k) for k in range(4)])
        all_semi_target = semi_target.repeat(8)
        
        
        non_negative_target = (all_semi_target != -1)

        shift_labels = torch.cat([torch.ones_like(semi_target) * k for k in range(4)], 0).to(device)  # B -> 4B
        shift_labels = shift_labels.repeat(2)

        images_pair = torch.cat([images1, images2], dim=0)  # 8B
        images_pair = simclr_aug(images_pair)
        
        _, outputs_aux = model(images_pair, simclr=True, penultimate=True, shift=True)
        out = F.normalize(outputs_aux['simclr'],dim=-1)
        pen_out = outputs_aux['shift']
        
        Ls = criterion(pen_out[non_negative_target, :], shift_labels[non_negative_target])
        
        score = energy_score(out, model)
        C = (torch.log(torch.Tensor([args.n_cluster])) + 1/args.temperature).to(device)
        Le = torch.where(all_semi_target == -1, (C - score) ** -1, score ** -1).mean()  
        L = Le + Ls #+ Le_shi
        optim.zero_grad()
        L.backward()
        optim.step()
        
        ## optimizer scheduler
        
        losses_energy.append(Le.cpu().detach())
        losses_shift.append(Ls.cpu().detach())

    # earlystop
    model.eval()
    with torch.no_grad():
        earlystop_auroc = earlystop_score(model,validation_dataset)
    earlystop_trace.append(earlystop_auroc)
    print('[{}]epoch loss:'.format(epoch), np.mean(losses_energy), np.mean(losses_shift))
    print('[{}]earlystop loss:'.format(epoch),earlystop_auroc)
    
    if max_earlystop_auroc < earlystop_auroc:
        max_earlystop_auroc = earlystop_auroc
        best_epoch = epoch
        best_model = copy.deepcopy(model)

    if (epoch % 3) == 0:
        model.eval()
        with torch.no_grad():
            print("redefine prototypes")
            model.module.prototypes = generate_prototypes(model, false_valid_loader, n_cluster=args.n_cluster)
    
print("best epoch:",best_epoch,"best auroc:",max_earlystop_auroc)
test(best_model, test_loader, train_loader, epoch) # we do not test them
checkpoint(best_model,  f'ckpt_ssl_{best_epoch}_best.pt', args, args.device)

