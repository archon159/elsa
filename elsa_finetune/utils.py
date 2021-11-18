import os
import logging
import torch


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
