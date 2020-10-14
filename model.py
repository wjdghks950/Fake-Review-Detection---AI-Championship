import torch
import torch.nn as nn
from kobert.pytorch_kobert import get_pytorch_kobert_model

class FakeDetector(nn.Module):
    def __init__(self, config):