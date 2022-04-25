
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from .pretrain import check_root, get_model, download, download_pretrain

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=os.environ.get("LOGLEVEL", "INFO"))


def fix_seed(seed=12345):
    import torch
    import numpy as np
    import random 
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) # gpu
    np.random.seed(seed) # numpy
    random.seed(seed) # random and transforms
    torch.backends.cudnn.deterministic=True # cudnn
