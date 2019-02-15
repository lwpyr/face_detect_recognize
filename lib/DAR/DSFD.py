import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
from PIL import ImageFont, ImageDraw, Image
import numpy as np

from ..dsfd.DSFD_vgg import build_net_vgg
from torch.autograd import Variable

class FaceD(object):
    def __init__(self, config):
        # size = config.SCALE.lower()
        # if size == "small":
        #     self.scale = [576, 1024]
        # elif size == "middle":
        #     self.scale = [864, 1536]
        # elif size == "big":
        #     self.scale = [1152, 2048]
        self.scale = config.SCALE
        self.mean = np.array([104., 117., 123.], dtype=np.float)[:, np.newaxis, np.newaxis]
        torch.no_grad()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(config.GPU_ID)
        self.net = build_net_vgg('test')
        self.net.load_state_dict(torch.load(config.MODEL_PATH))
        self.net.eval()
        self.net.cuda()
        cudnn.benckmark = True
        self.thresh = config.THRESH
        self.font = config.FONT_PATH

    def Detect(self, image):
        max_im_shrink = np.sqrt(
        float(self.scale[0]) * self.scale[1] / (image.shape[0] * image.shape[1]))
        img = cv2.resize(image, None, None, fx=max_im_shrink,
                           fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = img.astype('float32')
        img -= self.mean
        img = img[[2, 1, 0], :, :]
        x = Variable(torch.from_numpy(img).unsqueeze(0)).cuda()
        y = self.net(x)
        dets = y.data[0].reshape(-1, 5)
        keep = dets[:, 0] >= self.thresh
        dets = dets[keep, :]
        dets = dets[:, 1:] * torch.Tensor([[image.shape[1], image.shape[0], image.shape[1], image.shape[0]]])
        return dets.cpu().numpy()

    def Detect_raw(self, image):
        max_im_shrink = np.sqrt(
        200.0 * 400.0 / (image.shape[0] * image.shape[1]))
        img = cv2.resize(image, None, None, fx=max_im_shrink,
                           fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = img.astype('float32')
        img -= self.mean
        img = img[[2, 1, 0], :, :]
        x = Variable(torch.from_numpy(img).unsqueeze(0)).cuda()
        y = self.net(x)
        dets = y.data[0].reshape(-1, 5)
        keep = dets[:, 0] >= self.thresh
        dets = dets[keep, :]
        dets = dets[:, 1:] * torch.Tensor([[image.shape[1], image.shape[0], image.shape[1], image.shape[0]]])
        return dets.cpu().numpy()

    def reset(self):
        # PyTorch does not need *bind* operations
        pass

    def vis_dets(self, img, dets, names, scores=None):
        img = img.copy()
        num = len(dets)
        for idx, bbox in enumerate(dets):
            cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (127, 255, 0), 4)
            cv2.rectangle(img,(int(bbox[0]-2), int(bbox[1]-25)),(int(bbox[0]+100), int(bbox[1])),(255, 0, 0), -1)
            if scores is not None:
                cv2.putText(img, '%.3f' % scores[idx], (int(bbox[0]-2), int(bbox[1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1, lineType=8)
        font = ImageFont.truetype(self.font, 22)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        for idx, bbox in enumerate(dets):
            draw.text((int(bbox[0]), int(bbox[1]-22)),  names[idx].decode('utf8'), font = font, fill = (255 ,255 ,255 ,0))
        img = np.array(img_pil)
        #cv2.putText(img, 'person %d' % num, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0 ,255), thickness = 5, lineType = 8)
        return img
