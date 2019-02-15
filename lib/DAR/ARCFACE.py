from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cPickle
import numpy as np
import mxnet as mx
import cv2
from skimage import transform as trans
from matplotlib import pyplot as plt


class FaceR(object):
    def __init__(self, config):
        image_size = (112, 112)
        self.model = self.get_model(mx.gpu(config.GPU_ID), image_size, config.MODEL_PATH, 'fc1')
        self.image_size = image_size
        self.thresh = config.THRESH
        self.mode = config.STANDARD.lower()
        if self.mode == 'mse':
            self.thresh = -self.thresh
  
    def get_model(self, ctx, image_size, prefix, layer):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
        all_layers = sym.get_internals()
        sym = all_layers[layer+'_output'] 
        model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        return model

    def load_id_files(self, folder_path=None):
        if os.path.exists('person_embedding'):
            person_embedding = cPickle.load(open('person_embedding', 'rb'))
            self.db_person = []
            self.db_embedding = []
            for p, e in person_embedding.items():
                self.db_person.append(p)
                self.db_embedding.append(e)
            self.db_embedding = np.vstack(self.db_embedding)
        else:
            if folder_path is None:
                raise NotImplementedError
            files = os.listdir(folder_path)
            embedding = []
            for f in files:
                img = cv2.cvtColor(cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                img_mat = img[np.newaxis, :]
                img_mat = np.asarray(img_mat, np.float)
                img_mat = mx.nd.array(img_mat)
                db = mx.io.DataBatch(data=(img_mat,))
                self.model.forward(db, is_train=False)
                embedding.append(self.model.get_outputs()[0].asnumpy())
            embedding = np.vstack(embedding)
            embedding = embedding/np.linalg.norm(embedding,2,axis=1).reshape(-1,1)
            self.db_person = [i.split('.')[0] for i in files]
            pickle = dict(zip(self.db_person, embedding))
            cPickle.dump(pickle, open('person_embedding', 'wb'))
            self.db_embedding = embedding

    def Recognize(self, aligned_list, batch_size=8):
        embedding = []

        batch_idx = range(0, len(aligned_list), batch_size)
        batch_idx.append(len(aligned_list))
        
        for bid in range(len(batch_idx)-1):
            img_mat = np.asarray(aligned_list[batch_idx[bid]:batch_idx[bid+1]], np.float)
            img_mat = img_mat[:,:,:,[2,1,0]]
            img_mat = np.transpose(img_mat, [0, 3, 1, 2])
            img_mat = mx.nd.array(img_mat)
            db = mx.io.DataBatch(data=(img_mat,))
            self.model.forward(db, is_train=False)
            embedding.append(self.model.get_outputs()[0].asnumpy())

        embedding = np.vstack(embedding)
        embedding = embedding/np.linalg.norm(embedding,2, axis=1).reshape(-1,1)
        if self.mode == 'cosine':
            face_score = embedding.dot(self.db_embedding.T)
            face_score_argmax = face_score.argmax(axis=1)
            face_score_max = face_score[np.arange(len(face_score)), face_score_argmax]
        elif self.mode == 'mse':
            face_score_argmax = []
            face_score_max = []
            for f in embedding:
                scores = np.sum(np.square(f-self.db_embedding), axis=1)
                argmax = scores.argmin()
                face_score_argmax.append(argmax)
                face_score_max.append(-scores[argmax])
        name_list = []
        for i, score in zip(face_score_argmax, face_score_max):
            if score < self.thresh:
                name_list.append('null')
            else:
                name_list.append(self.db_person[i])
        return name_list, face_score_max

    def pre_test(self):
        face_confuix = self.db_embedding.dot(self.db_embedding.T)
        plt.imshow(face_confuix)
        self.confuix = face_confuix
        plt.show()