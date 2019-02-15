import yaml, os
import numpy as np
from easydict import EasyDict as edict

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

config = edict()

config.DAR_VERSION = 0.4
config.NEED_MXNET = True
config.NEED_PYTORCH = True

config.DATABASE_READY = True
config.ID_FOLDER = ""
config.ALIGNED_FOLDER = ""

config.DETECT = edict()
config.ALIGN = edict()
config.RECOGNIZE = edict()

config.DETECT_MODEL = "FPN"

config.DETECT.SCALE = [600, 1000]
config.DETECT.THRESH = 0.7
config.DETECT.SYMBOL_PATH = './lib/data/fpn_widerface.json'
config.DETECT.MODEL_PATH = ""
config.DETECT.GPU_ID = 0
config.DETECT.FIXSIZE = True
config.DETECT.FONT_PATH = "/home/liwen/src/face_DAR/lib/data/simsun.ttc"


config.ALIGN_MODEL = "ONET"

config.ALIGN.THRESH = 0.3
config.ALIGN.MODEL_PATH = "./lib/mtcnn/det3"
config.ALIGN.GPU_ID = 0


config.RECOGNIZE_MODEL = ""

config.RECOGNIZE.STANDARD = "MSE"
config.RECOGNIZE.THRESH = 1.24
config.RECOGNIZE.BATCH_SIZE = 8
config.RECOGNIZE.MODEL_PATH = ""
config.RECOGNIZE.GPU_ID = 0

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALE':
                        config[k] = list(v)
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py")