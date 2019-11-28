import os

import cv2
import numpy as np
from PIL import Image

from common.cfg import Config as cfg
from common.timeit_decorator import timeit
from crnn.image_util import union_rbox, rotate_cut_img, sort_box
from crnn.keys import alphabetChinese
from crnn.network_torch import CRNN


class CRNNRecognizer():
    def __init__(self):
        alphabet = alphabetChinese
        nclass = len(alphabet) + 1
        GPU = cfg.RUNNING_MODE == 'gpu'
        crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=cfg.CRNN_LSTMFLAG, GPU=GPU, alphabet=alphabet)
        if os.path.exists(cfg.CRNN_MODEL):
            crnn.load_weights(cfg.CRNN_MODEL)
        else:
            print("Failed to load CRNN model!")
            exit()

        self.crnn = crnn

    @timeit
    def recognize(self, img, boxes):
        im = Image.fromarray(img)
        boxes = sort_box(boxes)
        newBoxes = []
        for index, box in enumerate(boxes):
            partImg, box = rotate_cut_img(im, box, cfg.CRNN_LEFTADJUSTALPH, cfg.CRNN_RIGHTADJUSTALPH)
            cv2.imwrite(os.path.join(cfg.TEST_RESULT, 'XX_{}.jpg'.format(index)), np.array(partImg))

            box['img'] = partImg.convert('L')
            newBoxes.append(box)

        result = self.crnn.predict_job(newBoxes)
        result = union_rbox(result, 0.2)
        return result
