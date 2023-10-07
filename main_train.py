import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from torch import nn, optim
import random
import matplotlib
import matplotlib.pyplot as plt
from gymnasium.wrappers import FlattenObservation

from collections import namedtuple, deque
from itertools import count, chain

# import csv

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
print("device: ", device)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

