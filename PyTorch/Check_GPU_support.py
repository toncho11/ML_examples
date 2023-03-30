# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:08:59 2023

@author: antona
"""

import torch

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.cuda.current_device())


print(torch.cuda.get_device_name(0))