import os

import cv2
import numpy as np
from PIL import Image

from crnn.image_util import union_rbox, rotate_cut_img
from crnn.keys import alphabetChinese
from crnn.network_torch import CRNN

TEST_RESULT = './test_result'


class CRNNRecognizer():
    def __init__(self):
        alphabet = alphabetChinese
        nclass = len(alphabet) + 1
        LSTMFLAG = True
        GPU = True
        crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, GPU=GPU, alphabet=alphabet)
        ocrModelTorchLstm = os.path.join(os.getcwd(), "model", "ocr-lstm.pth")
        if os.path.exists(ocrModelTorchLstm):
            crnn.load_weights(ocrModelTorchLstm)
        else:
            print("download model or tranform model with tools!")
            exit()

        self.crnn = crnn

    def recognize(self, img, boxes):
        leftAdjustAlph, rightAdjustAlph = 0.01, 0.01

        im = Image.fromarray(img)
        newBoxes = []
        for index, box in enumerate(boxes):
            partImg, box = rotate_cut_img(im, box, leftAdjustAlph, rightAdjustAlph)

            cv2.imwrite(os.path.join(TEST_RESULT, 'XX_{}.jpg'.format(index)), np.array(partImg))

            box['img'] = partImg.convert('L')
            newBoxes.append(box)

        result = self.crnn.predict_job(newBoxes)
        result = union_rbox(result, 0.2)
        return result
