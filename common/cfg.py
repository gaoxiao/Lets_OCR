import numpy as np


class Config:
    TEST_RESULT = './test_result'
    MEAN = np.float32([102.9801, 115.9465, 122.7717])
    # MEAN=np.float32([100.0, 100.0, 100.0])
    TEST_GPU_ID = 0
    SCALE = 900
    MAX_SCALE = 1500
    TEXT_PROPOSALS_WIDTH = 0
    MIN_RATIO = 0.01
    LINE_MIN_SCORE = 0.6
    TEXT_LINE_NMS_THRESH = 0.3
    MAX_HORIZONTAL_GAP = 30
    TEXT_PROPOSALS_MIN_SCORE = 0.9
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MIN_NUM_PROPOSALS = 0
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6
    ANCHOR_HEIGHT = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]
    CTPN_MODEL = './model/ctpn-msra_ali-9-end.model'
    CRNN_MODEL = './model/ocr-lstm.pth'
    BOX_THRESH_HOLD = 0.7
    RUNNING_MODE = 'gpu'
    CRNN_LSTMFLAG = True
    CRNN_LEFTADJUSTALPH = 0.01
    CRNN_RIGHTADJUSTALPH = 0.01
