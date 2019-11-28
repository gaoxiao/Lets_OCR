import copy
import os
import time

import cv2
import numpy as np
import torch

import Net.net as Net
import lib.dataset_handler
import lib.draw_image
import lib.utils
from common.cfg import Config as cfg
from lib.nms_wrapper import nms
from text_proposal_connector import TextProposalConnector


class CTPNDetector():
    def __init__(self):

        ctpn = Net.CTPN()
        if cfg.RUNNING_MODE == 'cpu':
            ctpn.load_state_dict(torch.load(cfg.CTPN_MODEL, map_location=cfg.RUNNING_MODE))
        else:
            ctpn.load_state_dict(torch.load(cfg.CTPN_MODEL))
            ctpn = ctpn.cuda()
        print(ctpn)
        ctpn.eval()
        self.ctpn = ctpn

    def detect(self, im_name):
        im = cv2.imread(im_name)
        im = lib.dataset_handler.scale_img_only(im)
        img = copy.deepcopy(im)
        img = img.transpose(2, 0, 1)
        cpu_img = img[np.newaxis, :, :, :]
        img = torch.Tensor(cpu_img)

        tttt = time.time()

        img = img.cuda()

        v, score, side = self.ctpn(img, val=True)
        print("net takes time:{}s".format(time.time() - tttt))
        tttt = time.time()

        orig_shape = score.shape
        dim_2 = orig_shape[2]
        dim_1 = orig_shape[1] * dim_2
        flatten = score[:, :, :, 1].flatten()
        mask = flatten >= cfg.BOX_THRESH_HOLD
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
            pt = lib.utils.trans_to_2pt(box[1], box[0] * 16 + 7.5, cfg.ANCHOR_HEIGHT[int(box[2])])
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
        cv2.imwrite(os.path.join(cfg.TEST_RESULT, os.path.basename(basename)), out_im)

        for p in text_proposals:
            lib.draw_image.draw_box_2pt(out_im, p)
        cv2.imwrite(os.path.join(cfg.TEST_RESULT, 'XX_' + basename), out_im)

        return im, text_recs


def normalize(data):
    if data.shape[0] == 0:
        return data
    max_ = data.max()
    min_ = data.min()
    return (data - min_) / (max_ - min_) if max_ - min_ != 0 else data - min_


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
