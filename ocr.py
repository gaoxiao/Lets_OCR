# coding=utf-8
import os
import shutil
import time

from crnn.recongnizer import CRNNRecognizer
from ctpn.detector import CTPNDetector

IMG_ROOT = "./common/OCR_TEST"
TEST_RESULT = './test_result'
NMS_THRESH = 0.3
NEIGHBOURS_MIN_DIST = 50
MIN_ANCHOR_BATCH = 2
MODEL = './model/ctpn-msra_ali-9-end.model'


def ocr_one(im_name, detector, recognizer):
    img, boxes = detector.detect(im_name)
    result = recognizer.recognize(img, boxes)
    for r in result:
        print(r['text'])


if __name__ == '__main__':
    if os.path.exists(TEST_RESULT):
        shutil.rmtree(TEST_RESULT)
    detector = CTPNDetector()

    recognizer = CRNNRecognizer()

    os.mkdir(TEST_RESULT)

    # img_file = './test2.png'
    img_file = 'common/OCR_TEST/000452.jpg'
    tttt = time.time()
    ocr_one(img_file, detector, recognizer)
    print("It takes time:{}s".format(time.time() - tttt))
    print("---------------------------------------")
