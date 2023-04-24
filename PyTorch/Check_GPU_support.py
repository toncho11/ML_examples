# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:08:59 2023

@author: antona

A script that tests if CUDA is usuable by PyTorch
"""

import torch

print("Is CUDA available on PyTorch:", torch.cuda.is_available())

print("How many CUDA devices available: ",torch.cuda.device_count())

print("Current CUDA device ID (starts by 0):", torch.cuda.current_device())

print("CUDA device name:",torch.cuda.get_device_name(0))