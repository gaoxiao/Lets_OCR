# coding=utf-8
import copy
import os
import shutil
import time

import cv2
import numpy as np
import torch
from PIL import Image

import Net.net as Net
import lib.dataset_handler
import lib.draw_image
import lib.utils
from cfg import Config as cfg
from crnn.image_util import rotate_cut_img, union_rbox

from crnn.network_torch import CRNN
from lib.nms_wrapper import nms
from text_proposal_connector import TextProposalConnector
from crnn.keys import alphabetChinese, alphabetEnglish

anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]

IMG_ROOT = "../common/OCR_TEST"
TEST_RESULT = './test_result'
THRESH_HOLD = 0.7
NMS_THRESH = 0.3
NEIGHBOURS_MIN_DIST = 50
MIN_ANCHOR_BATCH = 2
MODEL = './model/ctpn-msra_ali-9-end.model'


def convert_to_4pts(bboxes):
    """
        boxes: bounding boxes
    """
    text_recs = np.zeros((len(bboxes), 8), np.int)

    b1 = bboxes[:, 6] - bboxes[:, 7] / 2
    b2 = bboxes[:, 6] + bboxes[:, 7] / 2

    text_recs[:, 0] = bboxes[:, 0]
    text_recs[:, 1] = bboxes[:, 5] * bboxes[:, 0] + b1
    text_recs[:, 2] = bboxes[:, 2]
    text_recs[:, 3] = bboxes[:, 5] * bboxes[:, 2] + b1
    text_recs[:, 4] = bboxes[:, 2]
    text_recs[:, 5] = bboxes[:, 5] * bboxes[:, 2] + b2
    text_recs[:, 6] = bboxes[:, 0]
    text_recs[:, 7] = bboxes[:, 5] * bboxes[:, 0] + b2

    return text_recs


def detect(im_name, ctpn):
    im = cv2.imread(im_name)
    im = lib.dataset_handler.scale_img_only(im)
    img = copy.deepcopy(im)
    img = img.transpose(2, 0, 1)
    cpu_img = img[np.newaxis, :, :, :]
    img = torch.Tensor(cpu_img)

    tttt = time.time()

    img = img.cuda()

    v, score, side = ctpn(img, val=True)
    print("net takes time:{}s".format(time.time() - tttt))
    tttt = time.time()

    orig_shape = score.shape
    dim_2 = orig_shape[2]
    dim_1 = orig_shape[1] * dim_2
    flatten = score[:, :, :, 1].flatten()
    mask = flatten >= THRESH_HOLD
    indices = torch.nonzero(mask).flatten()
    masked_scores = flatten[mask]
    new_result = torch.Tensor(masked_scores.shape[0], 4)
    new_result[:, 0] = (indices % dim_1 / dim_2).int()  # dim 2  j
    new_result[:, 1] = (indices % dim_2).int()  # dim 3  k
    new_result[:, 2] = (indices / dim_1).int()  # dim 1  i
    new_result[:, 3] = masked_scores

    print("new filter takes time:{}s".format(time.time() - tttt))
    tttt = time.time()
    result = new_result.cpu()

    proposals = []
    for box in result:
        pt = lib.utils.trans_to_2pt(box[1], box[0] * 16 + 7.5, anchor_height[int(box[2])])
        proposals.append([pt[0], pt[1], pt[2], pt[3]])
    proposals = np.array(proposals, dtype=np.float32)

    masked_scores = masked_scores.cpu().detach().numpy()
    masked_scores = normalize(masked_scores)
    masked_scores = masked_scores[:, np.newaxis]

    keep_inds = nms(np.hstack((proposals, masked_scores)), cfg.TEXT_PROPOSALS_NMS_THRESH)
    text_proposals, scores = proposals[keep_inds], masked_scores[keep_inds]

    print("nms takes time:{}s".format(time.time() - tttt))
    tttt = time.time()

    text_proposal_connector = TextProposalConnector()
    text_lines = text_proposal_connector.get_text_lines(text_proposals, scores, cpu_img.shape[2:])

    print("text_proposal_connector takes time:{}s".format(time.time() - tttt))

    text_recs = convert_to_4pts(text_lines)

    _, basename = os.path.split(im_name)

    out_im = im.copy()

    for box in text_recs:
        box = np.array(box)
        # print(box)
        lib.draw_image.draw_ploy_4pt(out_im, box, thickness=2)
    cv2.imwrite(os.path.join(TEST_RESULT, os.path.basename(basename)), out_im)

    for p in text_proposals:
        lib.draw_image.draw_box_2pt(out_im, p)
    cv2.imwrite(os.path.join(TEST_RESULT, 'XX_' + basename), out_im)

    return im, text_recs


def recognize_batch(crnn, img, boxes, leftAdjustAlph, rightAdjustAlph):
    im = Image.fromarray(img)
    newBoxes = []
    for index, box in enumerate(boxes):
        partImg, box = rotate_cut_img(im, box, leftAdjustAlph, rightAdjustAlph)

        cv2.imwrite(os.path.join(TEST_RESULT, 'XX_{}.jpg'.format(index)), np.array(partImg))

        box['img'] = partImg.convert('L')
        newBoxes.append(box)

    res = crnn(newBoxes)
    return res


def ocr_one(im_name, ctpn, crnn):
    img, boxes = detect(im_name, ctpn)

    leftAdjustAlph, rightAdjustAlph = 0.01, 0.01
    result = recognize_batch(crnn, img, boxes, leftAdjustAlph, rightAdjustAlph)

    result = union_rbox(result, 0.2)

    for r in result:
        print(r['text'])


def normalize(data):
    if data.shape[0] == 0:
        return data
    max_ = data.max()
    min_ = data.min()
    return (data - min_) / (max_ - min_) if max_ - min_ != 0 else data - min_


if __name__ == '__main__':
    # running_mode = sys.argv[2]  # cpu or gpu
    running_mode = 'gpu'  # cpu or gpu
    print("Mode: %s" % running_mode)
    ctpn = Net.CTPN()

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

    if running_mode == 'cpu':
        ctpn.load_state_dict(torch.load(MODEL, map_location=running_mode))
    else:
        ctpn.load_state_dict(torch.load(MODEL))
        ctpn = ctpn.cuda()
    print(ctpn)
    ctpn.eval()

    if os.path.exists(TEST_RESULT):
        shutil.rmtree(TEST_RESULT)

    os.mkdir(TEST_RESULT)

    # img_file = './test2.png'
    img_file = '../common/OCR_TEST/000452.jpg'
    tttt = time.time()
    ocr_one(img_file, ctpn, crnn.predict_job)
    print("It takes time:{}s".format(time.time() - tttt))
    print("---------------------------------------")
