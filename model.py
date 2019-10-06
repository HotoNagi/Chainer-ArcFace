#! /usr/bin/env python
# -*- conding:utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import reporter
from chainer import Variable
import cupy as cp
import math



# ひとまずここに書く（後でちゃんと直す予定）
m=0.40
s=30.0
easy_margin=False
cos_m = math.cos(m)
sin_m = math.sin(m)
th = math.cos(math.pi - m)
mm = math.sin(math.pi - m) * m
# ----------------------------------------

class VGG16Fine(chainer.Chain):
    def __init__(self, num_class=10):
        w = chainer.initializers.Normal(scale=0.01)
        super(VGG16Fine, self).__init__()

        with self.init_scope():
            self.base = L.VGG16Layers()
            self.weight = chainer.Parameter(chainer.initializers.Normal(scale=0.01), (num_class, 32))
            self.fc6 = L.Linear(None, 1024, initialW=w)
            self.fc7 = L.Linear(1024, 32, initialW=w)
            #self.fc8 = L.Linear(32, num_class)

    def __call__(self, x, t):
        h = self.base(x, layers=['conv5_3'])['conv5_3']
        self.cam = h
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(h, ratio=0.5)
        h = self.fc6(h)
        h = F.relu(h)

        h = F.dropout(h, ratio=0.5)
        h = self.fc7(h)
        h = F.relu(h)

        h = F.dropout(h, ratio=0.5)
        #h = self.fc8(h)

        ################################################################################
        #                           VGG16の後ろにArcFace実装
        ################################################################################
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(h), F.normalize(self.weight)) # fc8
        sine = F.sqrt(F.clip((1.0 - F.square(cosine)),0, 1))
        phi = cosine * cos_m - sine * sin_m
        if easy_margin:
            phi = F.where(cosine.data > 0, phi, cosine)
        else:
            phi = F.where(cosine.data > th, phi, cosine.data - mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = cp.eye(10)[t].astype(cp.float32)
        one_hot = Variable(one_hot)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = phi
        output *= s
        ################################################################################

        return output



class ArcFaceClassifer(L.Classifier):
    def __init__(self, predictor,
                 lossfun=F.softmax_cross_entropy,
                 accfun=F.accuracy,):
        super().__init__(predictor=predictor, lossfun=lossfun, accfun=accfun)
        self.predictor = predictor

    def forward(self, x, t):
        if not chainer.config.train:
            self.y = self.predictor(x, t)
            self.loss = self.lossfun(self.y, t)
        else:
            self.y= self.predictor(x, t)
            self.loss = self.lossfun(self.y, t)

        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss