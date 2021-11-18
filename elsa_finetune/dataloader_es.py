from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import random
import numpy as np

def load_dataset(data_path, normal_class, known_outlier_class, n_known_outlier_classes = 0,
                 ratio_known_normal = 0.0, ratio_known_outlier = 0.0, ratio_pollution = 0.0,
                 random_state=None,train_transform=None, test_transform=None,valid_transform=None):
    dataset = CIFAR10_Dataset(root=data_path,
                              normal_class=normal_class,
                              known_outlier_class=known_outlier_class,
                              n_known_outlier_classes=n_known_outlier_classes,
                              ratio_known_normal=ratio_known_normal,
                              ratio_known_outlier=ratio_known_outlier,
                              ratio_pollution=ratio_pollution,
                              train_transform=train_transform,
                              test_transform=test_transform,
                             valid_transform= valid_transform)
    return dataset


class CIFAR10_Dataset():
    def __init__(self, root, normal_class = [5], known_outlier_class = 3, n_known_outlier_classes = 0,
                 ratio_known_normal = 0.0, ratio_known_outlier = 0.0, ratio_pollution = 0.0, 
                 train_transform=None, test_transform=None,valid_transform=None):

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple(normal_class)
        self.outlier_classes = list(range(0, 10))
        for normal_cls in normal_class:
            self.outlier_classes.remove(normal_cls)
        self.outlier_classes = tuple(self.outlier_classes)
        self.root = root

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # CIFAR-10 preprocessing: feature scaling to [0, 1]
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyCIFAR10(root=self.root, train=True, transform=train_transform, target_transform=target_transform,
                              download=True)
        
        train_set.transform2 = valid_transform
        if valid_transform is not None:
            print("valid transform is activated")
        # Get valid set
        false_valid_set = MyCIFAR10(root=self.root, train=True, transform=test_transform, target_transform=target_transform,
                              download=True)
        

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels
        false_valid_set.semi_targets[idx] = torch.tensor(semi_targets) 
        
        #seperation from train set
        valset_ratio = 0.05
        valset_size = int(len(np.array(idx)[np.array(semi_targets)==0]) * valset_ratio)
        val_idx = list(np.random.choice(np.array(idx)[np.array(semi_targets)==0],size=valset_size,replace=False))
        
        train_idx = list(set(idx).difference(set(val_idx)))
        print('val dataset:',len(val_idx))
        print("train dataset:",len(train_idx))
        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, train_idx)
        self.valid_set = Subset(train_set, val_idx)
        self.false_valid_set = Subset(false_valid_set,idx)
        
        # Get test set
        self.test_set = MyCIFAR10(root=self.root, train=False, transform=test_transform, target_transform=target_transform,
                                  download=True)
        
    def loaders(self, batch_size, shuffle_train=True, shuffle_test=False, num_workers=4):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        false_valid_loader = DataLoader(dataset=self.false_valid_set, batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=num_workers, drop_last=False)        
        valid_loader = DataLoader(dataset=self.valid_set, batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=num_workers, drop_last=False)        
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, false_valid_loader,valid_loader, test_loader


class MyCIFAR10(CIFAR10):
    """
    Torchvision CIFAR10 class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)
        self.transform2 = None
        self.raw_tf = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], self.targets[index], int(self.semi_targets[index])
        cls = target
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        raw = self.raw_tf(img)
        if self.transform is not None:
            pos_1 = self.transform(img)
            if self.transform2 is not None:
                pos_2 = self.raw_tf(self.transform2(img))
            else:
                pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, semi_target, cls, raw


def create_semisupervised_setting(labels, normal_classes, outlier_classes, known_outlier_classes,
                                  ratio_known_normal, ratio_known_outlier, ratio_pollution):
    """
    Create a semi-supervised data setting. 
    :param labels: np.array with labels of all dataset samples
    :param normal_classes: tuple with normal class labels
    :param outlier_classes: tuple with anomaly class labels
    :param known_outlier_classes: tuple with known (labeled) anomaly class labels
    :param ratio_known_normal: the desired ratio of known (labeled) normal samples
    :param ratio_known_outlier: the desired ratio of known (labeled) anomalous samples
    :param ratio_pollution: the desired pollution ratio of the unlabeled data with unknown (unlabeled) anomalies.
    :return: tuple with list of sample indices, list of original labels, and list of semi-supervised labels
    """
    idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()
    idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()
    idx_known_outlier_candidates = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()

    n_normal = len(idx_normal)

    # Solve system of linear equations to obtain respective number of samples
    a = np.array([[1, 1, 0, 0],
                  [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                  [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                  [0, -ratio_pollution, (1-ratio_pollution), 0]])
    b = np.array([n_normal, 0, 0, 0])
    x = np.linalg.solve(a, b)

    # Get number of samples
    n_known_normal = int(x[0])
    n_unlabeled_normal = int(x[1])
    n_unlabeled_outlier = int(x[2])
    n_known_outlier = int(x[3])
    
    print("# of known normal: ", n_known_normal)
    print("# of known outlier: ", n_known_outlier)

    # Sample indices
    perm_normal = np.random.permutation(n_normal)
    perm_outlier = np.random.permutation(len(idx_outlier))
    perm_known_outlier = np.random.permutation(len(idx_known_outlier_candidates))

    idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()
    idx_unlabeled_normal = idx_normal[perm_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()
    idx_unlabeled_outlier = idx_outlier[perm_outlier[:n_unlabeled_outlier]].tolist()
    idx_known_outlier = idx_known_outlier_candidates[perm_known_outlier[:n_known_outlier]].tolist()

    # Get original class labels
    labels_known_normal = labels[idx_known_normal].tolist()
    labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
    labels_known_outlier = labels[idx_known_outlier].tolist()

    # Get semi-supervised setting labels
    semi_labels_known_normal = np.ones(n_known_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(np.int32).tolist()
    semi_labels_known_outlier = (-np.ones(n_known_outlier).astype(np.int32)).tolist()

    # Create final lists
    list_idx = idx_known_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_known_outlier
    list_labels = labels_known_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_known_outlier
    list_semi_labels = (semi_labels_known_normal + semi_labels_unlabeled_normal + semi_labels_unlabeled_outlier
                        + semi_labels_known_outlier)
    print("# of training set: ", len(list_idx))

    return list_idx, list_labels, list_semi_labels
