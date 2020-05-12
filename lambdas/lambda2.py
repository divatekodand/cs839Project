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


class Lambda2(nn.Module):

    def __init__(self, block_size, lstm_dim=256):
        super(Lambda2, self).__init__()
        self.out_size = out_size
        self.features = nn.Sequential(
           
        )

    def forward(self, x):
        x = self.features(x)
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
    model = Lambda2(1600)
    print(os.listdir('/opt/python/lib/python3.6/site-packages/model_layer000/'))
    ## model.load_state_dict(torch.load('/opt/python/lib/python3.6/site-packages/model0/model0.pt'))
    data_out = model.forward(data_tensor)
    r.set(new_data_key, pickle.dumps(data_out))
    return "Uploaded"
