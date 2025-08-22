apt -yq install swig >/dev/null
pip -q install "gymnasium[box2d]" imageio imageio-ffmpeg opencv-python-headless einops
import os, math, random, time
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.wrappers import RecordVideo