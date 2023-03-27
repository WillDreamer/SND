import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from unet import UNet
from utils import *

import os
import json




model = UNet(in_channels=3)
model.load_state_dict(torch.load(ckpt_fname))
