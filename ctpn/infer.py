# coding=utf-8
import copy
import math
import os
import random
import shutil
import time

import cv2
import lib.nms
import numpy as np
import torch

import Net.net as Net
import lib.dataset_handler
import lib.draw_image
import lib.utils
from cfg import Config as cfg
from lib.nms_wrapper import nms
from text_proposal_connector import TextProposalConnector

anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]

IMG_ROOT = "../common/OCR_TEST"
TEST_RESULT = './test_result'
THRESH_HOLD = 0.7
NMS_THRESH = 0.3
NEIGHBOURS_MIN_DIST = 50
MIN_ANCHOR_BATCH = 2
MODEL = './model/ctpn-msra_ali-9-end.model'


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1] - 1)
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0] - 1)
    return boxes


def fit_y(X, Y, x1, x2):
    len(X) != 0
    # if X only include one point, the function will get line y=Y[0]
    if np.sum(X == X[0]) == len(X):
        return Y[0], Y[0]
    p = np.poly1d(np.polyfit(X, Y, 1))
    return p(x1), p(x2)


def get_text_lines(text_proposals, im_size, scores=0):
    """
    text_proposals:boxes

    """
    # tp_groups = neighbour_connector(text_proposals, im_size)  # 首先还是建图，获取到文本行由哪几个小框构成
    # print(tp_groups)
    text_lines = np.zeros((len(text_proposals), 8), np.float32)

    for index, tp_indices in enumerate(text_proposals):
        text_line_boxes = np.array(tp_indices)  # 每个文本行的全部小框
        # print(text_line_boxes)
        # print(type(text_line_boxes))
        # print(text_line_boxes.shape)
        X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 求每一个小框的中心x，y坐标
        Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2
        # print(X)
        # print(Y)

        z1 = np.polyfit(X, Y, 1)  # 多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）

        x0 = np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
        x1 = np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值

        offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # 小框宽度的一半

        # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
        lt_y, rt_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
        # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
        lb_y, rb_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

        # score = scores[list(tp_indices)].sum() / float(len(tp_indices))  # 求全部小框得分的均值作为文本行的均值

        text_lines[index, 0] = x0
        text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
        text_lines[index, 2] = x1
        text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
        text_lines[index, 4] = scores  # 文本行得分
        text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
        text_lines[index, 6] = z1[1]
        height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
        text_lines[index, 7] = height + 2.5

    text_recs = np.zeros((len(text_lines), 9), np.float32)
    index = 0
    for line in text_lines:
        b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
        b2 = line[6] + line[7] / 2
        x1 = line[0]
        y1 = line[5] * line[0] + b1  # 左上
        x2 = line[2]
        y2 = line[5] * line[2] + b1  # 右上
        x3 = line[0]
        y3 = line[5] * line[0] + b2  # 左下
        x4 = line[2]
        y4 = line[5] * line[2] + b2  # 右下
        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

        fTmp0 = y3 - y1  # 文本行高度
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)  # 做补偿
        y = np.fabs(fTmp1 * disY / width)
        if line[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y
        # clock-wise order
        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x4
        text_recs[index, 5] = y4
        text_recs[index, 6] = x3
        text_recs[index, 7] = y3
        text_recs[index, 8] = line[4]
        index = index + 1

    text_recs = clip_boxes(text_recs, im_size)

    return text_recs


def meet_v_iou(y1, y2, h1, h2):
    def overlaps_v(y1, y2, h1, h2):
        return max(0, y2 - y1 + 1) / min(h1, h2)

    def size_similarity(h1, h2):
        return min(h1, h2) / max(h1, h2)

    return overlaps_v(y1, y2, h1, h2) >= 0.6 and size_similarity(h1, h2) >= 0.6


def gen_test_images(img_root, test_num=10):
    img_list = os.listdir(img_root)
    if test_num > 0:
        random_list = random.sample(img_list, test_num)
    else:
        random_list = img_list
    test_pair = []
    for im in random_list:
        name, _ = os.path.splitext(im)
        im_path = os.path.join(img_root, im)
        test_pair.append(im_path)
    return test_pair


def get_anchor_h(anchor, v):
    vc = v[int(anchor[7]), 0, int(anchor[5]), int(anchor[6])]
    vh = v[int(anchor[7]), 1, int(anchor[5]), int(anchor[6])]
    cya = anchor[5] * 16 + 7.5
    ha = anchor_height[int(anchor[7])]
    cy = vc * ha + cya
    h = math.pow(10, vh) * ha
    return h


def get_successions(v, anchors=[]):
    texts = []

    tttt = time.time()
    print("get_successions 0 takes time:{}s".format(time.time() - tttt))
    tttt = time.time()

    for i, anchor in enumerate(anchors):
        neighbours = []
        neighbours.append(i)
        center_x1 = (anchor[2] + anchor[0]) / 2
        h1 = get_anchor_h(anchor, v)
        # find i's neighbour
        for j in range(0, len(anchors)):
            if j == i:
                continue
            center_x2 = (anchors[j][2] + anchors[j][0]) / 2
            h2 = get_anchor_h(anchors[j], v)
            if abs(center_x1 - center_x2) < NEIGHBOURS_MIN_DIST and \
                    meet_v_iou(max(anchor[1], anchors[j][1]), min(anchor[3], anchors[j][3]), h1,
                               h2):  # less than 50 pixel between each anchor
                neighbours.append(j)
        if len(neighbours) != 0:
            texts.append(neighbours)
    print("get_successions 1 takes time:{}s".format(time.time() - tttt))
    tttt = time.time()
    need_merge = True
    while need_merge:
        need_merge = False
        # ok, we combine again.
        for i, line in enumerate(texts):
            if len(line) == 0:
                continue
            for index in line:
                for j in range(i + 1, len(texts)):
                    if index in texts[j]:
                        texts[i] += texts[j]
                        texts[i] = list(set(texts[i]))
                        texts[j] = []
                        need_merge = True
    print("get_successions 2 takes time:{}s".format(time.time() - tttt))
    tttt = time.time()
    result = []
    # print(texts)
    for text in texts:
        if len(text) < MIN_ANCHOR_BATCH:
            continue
        local = []
        for j in text:
            local.append(anchors[j])
        result.append(local)
    print("get_successions 3 takes time:{}s".format(time.time() - tttt))
    return result


def infer_one(im_name, net):
    im = cv2.imread(im_name)
    im = lib.dataset_handler.scale_img_only(im)
    img = copy.deepcopy(im)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    img = torch.Tensor(img)

    tttt = time.time()

    img = img.cuda()

    v, score, side = net(img, val=True)
    print("net takes time:{}s".format(time.time() - tttt))
    tttt = time.time()

    score = score.cpu()
    result = []
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            for k in range(score.shape[2]):
                if score[i, j, k, 1] > THRESH_HOLD:
                    result.append((j, k, i, float(score[i, j, k, 1].detach().numpy())))

    print("filter takes time:{}s".format(time.time() - tttt))
    tttt = time.time()

    for_nms = []
    for box in result:
        pt = lib.utils.trans_to_2pt(box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
        for_nms.append([pt[0], pt[1], pt[2], pt[3], box[3], box[0], box[1], box[2]])
    for_nms = np.array(for_nms, dtype=np.float32)
    nms_result = lib.nms.cpu_nms(for_nms, NMS_THRESH)

    out_nms = []
    for i in nms_result:
        out_nms.append(for_nms[i, 0:8])

    print('out_nms size: {}'.format(len(out_nms)))
    connect = get_successions(v, out_nms)
    print("get_successions takes time:{}s".format(time.time() - tttt))
    tttt = time.time()
    texts = get_text_lines(connect, im.shape)

    for box in texts:
        box = np.array(box)
        # print(box)
        lib.draw_image.draw_ploy_4pt(im, box[0:8], thickness=2)

    _, basename = os.path.split(im_name)
    cv2.imwrite(os.path.join(TEST_RESULT, basename), im)

    for i in nms_result:
        lib.draw_image.draw_box_2pt(im, for_nms[i, 0:4])
    cv2.imwrite(os.path.join(TEST_RESULT, 'XX_' + basename), im)
    _, basename = os.path.split(im_name)


def random_test(net):
    test_pair = gen_test_images(IMG_ROOT, 0)
    print(test_pair)

    for t in test_pair:
        im = cv2.imread(t)
        im = lib.dataset_handler.scale_img_only(im)
        img = copy.deepcopy(im)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :, :, :]
        img = torch.Tensor(img)
        v, score, side = net(img, val=True)
        result = []
        for i in range(score.shape[0]):
            for j in range(score.shape[1]):
                for k in range(score.shape[2]):
                    if score[i, j, k, 1] > THRESH_HOLD:
                        result.append((j, k, i, float(score[i, j, k, 1].detach().numpy())))

        for_nms = []
        for box in result:
            pt = lib.utils.trans_to_2pt(box[1], box[0] * 16 + 7.5, anchor_height[box[2]])
            for_nms.append([pt[0], pt[1], pt[2], pt[3], box[3], box[0], box[1], box[2]])
        for_nms = np.array(for_nms, dtype=np.float32)
        nms_result = lib.nms.cpu_nms(for_nms, NMS_THRESH)

        out_nms = []
        for i in nms_result:
            out_nms.append(for_nms[i, 0:8])

        connect = get_successions(v, out_nms)
        texts = get_text_lines(connect, im.shape)

        for i in nms_result:
            lib.draw_image.draw_box_2pt(im, for_nms[i, 0:4])
            # im = other.draw_box_h_and_c(im, int(for_nms[i, 6]), cy, h)

        for box in texts:
            box = np.array(box)
            print(box)
            lib.draw_image.draw_ploy_4pt(im, box[0:8], thickness=2)
        cv2.imwrite(os.path.join(TEST_RESULT, os.path.basename(t)), im)


"""
        for i in nms_result:
            vc = v[int(for_nms[i, 7]), 0, int(for_nms[i, 5]), int(for_nms[i, 6])]
            vh = v[int(for_nms[i, 7]), 1, int(for_nms[i, 5]), int(for_nms[i, 6])]
            cya = for_nms[i, 5] * 16 + 7.5
            ha = anchor_height[int(for_nms[i, 7])]
            cy = vc * ha + cya
            h = math.pow(10, vh) * ha
            lib.draw_image.draw_box_2pt(im, for_nms[i, 0:4])
            #im = other.draw_box_h_and_c(im, int(for_nms[i, 6]), cy, h)

        # print(result)
        #for box in result:
         #   im = other.draw_box_h_and_c(im, box[1], box[0] * 16 + 7.5, anchor_height[box[2]])

        for gt_box in gt:
            im = lib.draw_image.draw_box_4pt(im, gt_box, (255, 0, 0))

        cv2.imwrite(os.path.join(TEST_RESULT, os.path.basename(t[0])), im)
"""


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


def infer_one_test(im_name, net):
    im = cv2.imread(im_name)
    im = lib.dataset_handler.scale_img_only(im)
    img = copy.deepcopy(im)
    img = img.transpose(2, 0, 1)
    cpu_img = img[np.newaxis, :, :, :]
    img = torch.Tensor(cpu_img)

    tttt = time.time()

    img = img.cuda()

    v, score, side = net(img, val=True)
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
    tttt = time.time()

    # keep_inds = self.filter_boxes(text_lines)
    # text_lines = text_lines[keep_inds]
    #
    # if text_lines.shape[0] != 0:
    #     keep_inds = nms(text_lines, cfg.TEXT_LINE_NMS_THRESH)
    #     text_lines = text_lines[keep_inds]
    #
    # return text_lines

    text_recs = convert_to_4pts(text_lines)

    _, basename = os.path.split(im_name)

    for box in text_recs:
        box = np.array(box)
        # print(box)
        lib.draw_image.draw_ploy_4pt(im, box, thickness=2)
    cv2.imwrite(os.path.join(TEST_RESULT, os.path.basename(basename)), im)

    for p in text_proposals:
        lib.draw_image.draw_box_2pt(im, p)
    cv2.imwrite(os.path.join(TEST_RESULT, 'XX_' + basename), im)


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
    net = Net.CTPN()
    if running_mode == 'cpu':
        net.load_state_dict(torch.load(MODEL, map_location=running_mode))
    else:
        net.load_state_dict(torch.load(MODEL))
        net = net.cuda()
    print(net)
    net.eval()

    if os.path.exists(TEST_RESULT):
        shutil.rmtree(TEST_RESULT)

    os.mkdir(TEST_RESULT)

    # if sys.argv[1] == 'random':
    #     random_test(net)
    # else:
    #     url = sys.argv[1]
    #     infer_one(url, net)

    # import time
    # root = '../common/OCR_TEST'
    # for p in os.listdir(root)[:5]:
    #     if not p.endswith('jpg'):
    #         continue
    #     p = os.path.join(root, p)
    #     tttt = time.time()
    #     infer_one_test(p, net)
    #     print("It takes time:{}s".format(time.time() - tttt))
    #     print("---------------------------------------")

    img_file = './test2.png'
    # img_file = '../common/OCR_TEST/000017.jpg'
    tttt = time.time()
    # infer_one(img_file, net)
    infer_one_test(img_file, net)
    print("It takes time:{}s".format(time.time() - tttt))
    print("---------------------------------------")
