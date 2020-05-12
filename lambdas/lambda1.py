try:
    import unzip_requirements
except ImportError:
    pass

import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import boto3
import logging
import redis
import pickle
import re
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class TFiLMSmallCNN(nn.Module):
    """
    TFiLMSmallCNN model for text classification
    """

    def __init__(self, block_size, out_size, lstm_dim=256):
        super(TFiLMSmallCNN, self).__init__()
        self.out_size = out_size
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True),
            TFilM(block_size, 0), 
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True),
            TFilM(block_size, 0), 
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True),
            TFilM(block_size, 0)
        )
        self.fc = nn.Linear(out_size, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x.view(x.shape[0], self.out_size))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Lambda1(nn.Module):

    def __init__(self, block_size, lstm_dim=256):
        super(Lambda1, self).__init__()
        self.fc = nn.Linear(out_size, 1)

    def forward(self, x):
        x = self.fc(x.view(x.shape[0], self.out_size))
        return x

def next_level(match):
    match = match.group()
    return str(int(match) + 1)

def lambda_handler(event, context):
    r = redis.StrictRedis(host='redis.me7jbk.ng.0001.use2.cache.amazonaws.com', port=6379, db=0)
    logger.info(event['key'])
    data_key = event['key']
    new_data_key = re.sub('[0-9]+', next_level, data_key)
    data = r.get(data_key)
    data_tensor = pickle.loads(data)
    model = Lambda1(1600)
    print(os.listdir('/opt/python/lib/python3.6/site-packages/model_layer000/'))
    ## model.load_state_dict(torch.load('/opt/python/lib/python3.6/site-packages/model0/model0.pt'))
    data_out = model.forward(data_tensor)
    r.set(new_data_key, pickle.dumps(data_out))
    return "Uploaded"
