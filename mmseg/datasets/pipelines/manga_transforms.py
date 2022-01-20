# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from ..builder import PIPELINES

@PIPELINES.register_module()
class Invert(object):

    def __init__(self, prob=None):
        self.prob = prob
        if prob is not None:
            assert prob >= 0 and prob <= 1

    def __call__(self, results):
        if np.random.rand() < self.prob:
            img = results['img']
            white = np.full_like(img, 255)
            results['img'] = white - img
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'