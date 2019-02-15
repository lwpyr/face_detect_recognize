import cv2
import os
import sys
import time
import mxnet as mx
from mxnet.module import Module
from ..nms.nms import py_nms_wrapper
from ..fpn.pyramid_proposal import *
from ..fpn.fpn_roi_pooling import *
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def transform(im, pixel_means):
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor

def resize(im, scale, stride=32):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(scale[0]) / float(im_size_min)
    if np.round(im_scale * im_size_max) > scale[1]:
        im_scale = float(scale[1]) / float(im_size_max)
    im = cv2.resize(im, (int(im.shape[1]*im_scale), int(im.shape[0]*im_scale)), interpolation=cv2.INTER_CUBIC)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def load_param(prefix, epoch, convert=False, ctx=None, process=False):
    arg_params, aux_params = load_checkpoint(prefix, epoch)
    if process:
        tests = [k for k in arg_params.keys() if '_test' in k]
        for test in tests:
            arg_params[test.replace('_test', '')] = arg_params.pop(test)
    return arg_params, aux_params

def load_checkpoint(prefix, epoch):
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params




class FaceD(object):
    def __init__(self, config):
        # size = config.SCALE.lower()
        # if size == "small":
        #     scale = [576, 1024]
        # elif size == "middle":
        #     scale = [864, 1536]
        # elif size == "big":
        #     scale = [1152, 2048]    
        sym = mx.sym.load(config.SYMBOL_PATH)
        
        self.nms = py_nms_wrapper(0.3)
        self.scale = config.SCALE
        self.mod = Module(sym, ['data', 'im_info'], [], context=[mx.gpu(config.GPU_ID)])
        self.thresh = config.THRESH
        self.rebind = not config.FIXSIZE
        self.model_path = config.MODEL_PATH
        self.font = config.FONT_PATH
        self.preprocess = False

    def bbox_detect(self, im, im_scale, force_rebind=False):
    
        im_tensor = transform(im, [103.06, 115.9, 123.15])
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)

        data = [mx.nd.array(im_tensor), mx.nd.array(im_info)]
        data_batch = mx.io.DataBatch(data=data, label=[], pad=0, index=0,
                                    provide_data=[[(k, v.shape) for k, v in zip(self.mod.data_names, data)]],
                                    provide_label=[None])

        if not self.mod.binded:
            arg_params, aux_params = load_param(self.model_path, 0, process=True)
            self.mod.bind([('data', (1L, 3L, im_tensor.shape[2], im_tensor.shape[3])), ('im_info', (1L, 3L))], None, 
                        for_training=False, inputs_need_grad=False, force_rebind=True,
                        shared_module=None)
            self.mod.init_params(arg_params=arg_params, aux_params=aux_params)
        
        if self.rebind or force_rebind:
            self.mod.bind([('data', (1L, 3L, im_tensor.shape[2], im_tensor.shape[3])), ('im_info', (1L, 3L))], None, 
                          for_training=False, inputs_need_grad=False, force_rebind=True,
                          shared_module=None)

        scale = data_batch.data[1].asnumpy()[0, 2]
        self.mod.forward(data_batch)
        output=dict(zip(self.mod.output_names, tuple(self.mod.get_outputs(merge_multi_context=False))))

        rois = output['rois_output'][0].asnumpy()[:, 1:]
        im_shape = data[0].shape

        scores = output['cls_prob_reshape_output'][0].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'][0].asnumpy()[0]

        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

        pred_boxes = pred_boxes / scale

        pred_boxes = pred_boxes.astype('f')
        scores = scores.astype('f')
        
        indexes = np.where(scores[:, 1] > self.thresh)[0]
        cls_scores = scores[indexes, 1, np.newaxis]
        cls_boxes = pred_boxes[indexes, 4:8]
        cls_dets = np.hstack((cls_boxes, cls_scores))
        keep = self.nms(cls_dets)
        return cls_dets[keep, :]

    def Detect(self, img):
        im, im_scale = resize(img, self.scale)
        dets = self.bbox_detect(im, im_scale)
        return dets
    
    def Detect_raw(self, img):
        im, im_scale = resize(img, [200, 400])
        dets = self.bbox_detect(im, im_scale, True)
        return dets

    def reset(self):
        self.mod.binded = False

    def vis_detections(self, img, dets, save='./tmp.jpg'):
        for bbox in dets:
            cv2.rectangle(img,(bbox[0], bbox[1]),(bbox[2],bbox[3]),(127, 255, 0), 4)
        cv2.imwrite(save, img)

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
