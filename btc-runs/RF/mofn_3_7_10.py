#!/usr/bin/env python3
#
# This code has been produced by an enterprise version of Brainome(tm) licensed to: andy Stevko.
# Portions of this code copyright (c) 2019-2022 by Brainome, Inc. All Rights Reserved.
# Distribution and use of this code or commercial use is permitted within the license terms
# set forth in a written contractual agreement between Brainome, Inc and brainome-user.
# Please contact support@brainome.ai with any questions.
# Use of predictions results at your own risk.
#
# Output of Brainome v1.8-120-prod.
# Invocation: brainome TRAIN_TEST_SPLITS/mofn-3-7-10-clean-train.csv -f RF -y -split 70 -modelonly -q -o btc-runs/RF/mofn_3_7_10.py -json btc-runs/RF/mofn_3_7_10.json
# Total compiler execution time: 0:00:06.03. Finished on: Feb-26-2022 18:34:55.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/RF/mofn_3_7_10.py
    Classifier Type:              Random Forest
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        77.97%
      Training accuracy:         100.00% (647/647 correct)
      Validation Accuracy:       100.00% (279/279 correct)
      Combined Model Accuracy:   100.00% (926/926 correct)


    Model Capacity (MEC):         32    bits
    Generalization Ratio:         15.34 bits/bit
    Percent of Data Memorized:    14.43%
    Resilience to Noise:          -1.30 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |  505    0 
                   1 |    0  142 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |  217    0 
                   1 |    0   62 

    Training Accuracy by Class:
               class |   TP   FP   TN   FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ---- ---- ---- ---- -------- -------- -------- -------- -------- --------
                   0 |  505    0  142    0  100.00%  100.00%  100.00%  100.00%  100.00%  100.00%
                   1 |  142    0  505    0  100.00%  100.00%  100.00%  100.00%  100.00%  100.00%

    Validation Accuracy by Class:
               class |   TP   FP   TN   FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ---- ---- ---- ---- -------- -------- -------- -------- -------- --------
                   0 |  217    0   62    0  100.00%  100.00%  100.00%  100.00%  100.00%  100.00%
                   1 |   62    0  217    0  100.00%  100.00%  100.00%  100.00%  100.00%  100.00%


    Attribute Ranking:
                                      Feature | Relative Importance
                                        Bit_8 :   0.1723
                                        Bit_3 :   0.1655
                                        Bit_4 :   0.1568
                                        Bit_6 :   0.1408
                                        Bit_7 :   0.1354
                                        Bit_2 :   0.1134
                                        Bit_5 :   0.1078
                                        Bit_9 :   0.0077
                                        Bit_1 :   0.0002
         

"""

import sys
import math
import argparse
import csv
import binascii
import faulthandler
import json
try:
    import numpy as np  # For numpy see: http://numpy.org
except ImportError as e:
    print("This predictor requires the Numpy library. Please run 'python3 -m pip install numpy'.", file=sys.stderr)
    raise e
try:
    from scipy.sparse import coo_matrix
    report_cmat = True
except ImportError:
    print("Note: If you install scipy (https://www.scipy.org) this predictor generates a confusion matrix. Try 'python3 -m pip install scipy'.", file=sys.stderr)
    report_cmat = False
try:
    import multiprocessing
    var_dict = {}
    default_to_serial = False
except:
    default_to_serial = True

IOBUF = 100000000
sys.setrecursionlimit(1000000)
TRAINFILE = ['TRAIN_TEST_SPLITS/mofn-3-7-10-clean-train.csv']
mapping = {'0': 0, '1': 1}
ignorelabels = []
ignorecolumns = []
target = ''
target_column = 10
important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ignore_idxs = []
classifier_type = 'RF'
num_attr = 10
n_classes = 2
model_cap = 32
logits_dict = {0: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.593999386, -0.279137909, 0.275892109, 0.226383552, 0.573168218, 0.240663409, 0.571018517]), 1: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.593999386, 0.279137909, -0.275892109, -0.226383552, -0.573168218, -0.240663409, -0.571018517]), 2: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.469867706, -0.428035051, 0.0799482167, 0.07985688, 0.392454416, 0.03750588, 0.366110712]), 3: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.469867706, 0.428035051, -0.0799482539, -0.0798568949, -0.392454386, -0.0375058949, -0.366110712]), 4: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.369013935, -0.40278998, 0.00726171071, 0.00737418374, 0.345618278, 0.0463879183, 0.307931542]), 5: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.369013935, 0.40278998, -0.00726172561, -0.00737420795, -0.345618278, -0.0463879108, -0.307931572]), 6: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.342692405, -0.404442936, -0.0227190889, -0.00878315512, 0.289078981, 0.0290248133, 0.333745301]), 7: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.342692405, 0.404442996, 0.022719115, 0.00878317468, -0.289078981, -0.0290248264, -0.333745301]), 8: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.326386094, -0.354988664, -0.0481665134, -0.0175270326, 0.271738738, -0.0112695424, 0.295941085]), 9: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.326386094, 0.354988664, 0.0481665246, 0.0175269917, -0.271738708, 0.011269547, -0.295941085]), 10: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.319973826, -0.315463185, -0.055367209, -0.0414326042, 0.250464618, -0.0263841823, 0.22778143]), 11: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.319973826, 0.315463215, 0.055367209, 0.0414326452, -0.250464648, 0.0263842084, -0.22778146]), 12: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30840832, -0.254296601, -0.00992530491, -0.0263242591, 0.231076881, -0.0270361137, 0.236491308]), 13: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.30840832, 0.254296571, 0.00992527232, 0.0263242517, -0.231076881, 0.0270360708, -0.236491293]), 14: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.29520306, -0.0599274598, -0.0504126735, 0.241973057, 0.00468567666, 0.207919568, 0.32539022, 0.175321802]), 15: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29520306, 0.0599274784, 0.050412681, -0.241973057, -0.00468571344, -0.207919568, -0.32539022, -0.175321817]), 16: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.276285976, -0.290110976, -0.0396807045, -0.0331832021, 0.221799374, -0.032704927, 0.217056215]), 17: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.276285946, 0.290110976, 0.0396807045, 0.0331831984, -0.221799374, 0.0327049568, -0.217056289]), 18: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.255501777, -0.267431349, -0.0394006073, -0.00111237459, 0.261892706, -0.0239441954, 0.171691522]), 19: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.255501777, 0.267431349, 0.0394005999, 0.00111237273, -0.261892676, 0.0239441246, -0.171691507]), 20: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.241006374, -0.241408348, -0.0306283291, -0.0396177396, 0.17619957, -0.0300652515, 0.195640191]), 21: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.241006359, 0.241408333, 0.0306283385, 0.0396177657, -0.17619957, 0.0300652534, -0.195640221]), 22: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.251529515, -0.226328045, -0.00846371707, -0.0378990099, 0.159836993, -0.0358011164, 0.206062526]), 23: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.251529515, 0.226328075, 0.00846372265, 0.0378990285, -0.159836948, 0.0358011313, -0.206062496])}
right_children_dict = {0: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 1: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 2: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 3: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 4: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 5: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 6: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 7: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 8: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 9: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 10: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 11: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 12: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 13: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 14: np.array([1, 3, 5, 7, 9, 11, 13, -1, -1, -1, -1, -1, -1, -1, -1]), 15: np.array([1, 3, 5, 7, 9, 11, 13, -1, -1, -1, -1, -1, -1, -1, -1]), 16: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 17: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 18: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 19: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 20: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 21: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 22: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1]), 23: np.array([1, 3, 5, 7, 9, 11, -1, -1, -1, -1, -1, -1, -1])}
split_feats_dict = {0: np.array([8, 3, 7, 4, 7, 4, 0, 0, 0, 0, 0, 0, 0]), 1: np.array([8, 3, 7, 4, 7, 4, 0, 0, 0, 0, 0, 0, 0]), 2: np.array([6, 2, 5, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0]), 3: np.array([6, 2, 5, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0]), 4: np.array([5, 7, 7, 8, 8, 3, 0, 0, 0, 0, 0, 0, 0]), 5: np.array([5, 7, 7, 8, 8, 3, 0, 0, 0, 0, 0, 0, 0]), 6: np.array([4, 6, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0, 0]), 7: np.array([4, 6, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0, 0]), 8: np.array([3, 5, 8, 7, 7, 4, 0, 0, 0, 0, 0, 0, 0]), 9: np.array([3, 5, 8, 7, 7, 4, 0, 0, 0, 0, 0, 0, 0]), 10: np.array([6, 8, 4, 2, 5, 2, 0, 0, 0, 0, 0, 0, 0]), 11: np.array([6, 8, 4, 2, 5, 2, 0, 0, 0, 0, 0, 0, 0]), 12: np.array([7, 4, 3, 3, 2, 8, 0, 0, 0, 0, 0, 0, 0]), 13: np.array([7, 4, 3, 3, 2, 8, 0, 0, 0, 0, 0, 0, 0]), 14: np.array([5, 6, 2, 8, 8, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0]), 15: np.array([5, 6, 2, 8, 8, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0]), 16: np.array([4, 2, 7, 5, 3, 3, 0, 0, 0, 0, 0, 0, 0]), 17: np.array([4, 2, 7, 5, 3, 3, 0, 0, 0, 0, 0, 0, 0]), 18: np.array([8, 7, 6, 6, 3, 5, 0, 0, 0, 0, 0, 0, 0]), 19: np.array([8, 7, 6, 6, 3, 5, 0, 0, 0, 0, 0, 0, 0]), 20: np.array([2, 3, 4, 4, 5, 5, 0, 0, 0, 0, 0, 0, 0]), 21: np.array([2, 3, 4, 4, 5, 5, 0, 0, 0, 0, 0, 0, 0]), 22: np.array([7, 3, 6, 8, 2, 8, 0, 0, 0, 0, 0, 0, 0]), 23: np.array([7, 3, 6, 8, 2, 8, 0, 0, 0, 0, 0, 0, 0])}
split_vals_dict = {0: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 4: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 5: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 6: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 7: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 8: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 9: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 10: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 11: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 12: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 13: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 14: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 15: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 16: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 17: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 18: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 19: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 20: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 21: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 22: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 23: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}


class PredictorError(Exception):

    def __init__(self, msg, code):
        self.msg = msg
        self.code = code

    def __str__(self):
        return self.msg

def __convert(cell):
    value = str(cell)
    try:
        result = int(value)
        return result
    except ValueError:
        try:
            result = float(value)
            if math.isnan(result):
                raise PredictorError('NaN value found. Aborting.', code=1)
            return result
        except ValueError:
            result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
            return result
        except Exception as e:
            raise e


def __get_key(val, dictionary):
    if dictionary == {}:
        return val
    for key, value in dictionary.items():
        if val == value:
            return key
    if val not in dictionary.values():
        raise PredictorError(f"Label {val} key does not exist", code=2)


def __confusion_matrix(y_true, y_pred, json):
    stats = {}
    labels = np.array(list(mapping.keys()))
    sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    for class_i in range(n_classes):
        class_i_label = __get_key(class_i, mapping)
        stats[int(class_i)] = {}
        class_i_indices = np.argwhere(y_true == class_i_label)
        not_class_i_indices = np.argwhere(y_true != class_i_label)
        # None represents N/A in this case
        stats[int(class_i)]['TP'] = TP = int(np.sum(y_pred[class_i_indices] == class_i_label)) if class_i_indices.size > 0 else None
        stats[int(class_i)]['FN'] = FN = int(np.sum(y_pred[class_i_indices] != class_i_label)) if class_i_indices.size > 0 else None
        stats[int(class_i)]['TN'] = TN = int(np.sum(y_pred[not_class_i_indices] != class_i_label)) if not_class_i_indices.size > 0 else None
        stats[int(class_i)]['FP'] = FP = int(np.sum(y_pred[not_class_i_indices] == class_i_label)) if not_class_i_indices.size > 0 else None
        if TP is None or FN is None or (TP + FN == 0):
            stats[int(class_i)]['TPR'] = None
        else:
            stats[int(class_i)]['TPR'] = (TP / (TP + FN))
        if TN is None or FP is None or (TN + FP == 0):
            stats[int(class_i)]['TNR'] = None
        else:
            stats[int(class_i)]['TNR'] = (TN / (TN + FP))
        if TP is None or FP is None or (TP + FP == 0):
            stats[int(class_i)]['PPV'] = None
        else:
            stats[int(class_i)]['PPV'] = (TP / (TP + FP))
        if TN is None or FN is None or (TN + FN == 0):
            stats[int(class_i)]['NPV'] = None
        else:
            stats[int(class_i)]['NPV'] = (TN / (TN + FN))
        if TP is None or FP is None or FN is None or (TP + FP + FN == 0):
            stats[int(class_i)]['F1'] = None
        else:
            stats[int(class_i)]['F1'] = ((2 * TP) / (2 * TP + FP + FN))
        if TP is None or FP is None or FN is None or (TP + FP + FN == 0):
            stats[int(class_i)]['TS'] = None
        else:
            stats[int(class_i)]['TS'] = (TP / (TP + FP + FN))

    if not report_cmat:
        return np.array([]), stats

    label_to_ind = {label: i for i, label in enumerate(labels)}
    y_pred = np.array([label_to_ind.get(x, n_classes + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_classes + 1) for x in y_true])

    ind = np.logical_and(y_pred < n_classes, y_true < n_classes)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    sample_weight = sample_weight[ind]

    cm = coo_matrix((sample_weight, (y_true, y_pred)), shape=(n_classes, n_classes), dtype=np.int64).toarray()
    with np.errstate(all='ignore'):
        cm = np.nan_to_num(cm)

    return cm, stats


def __preprocess_and_clean_in_memory(arr):
    clean_arr = np.zeros((len(arr), len(important_idxs)))
    for i, row in enumerate(arr):
        try:
            row_used_cols_only = [row[i] for i in important_idxs]
        except IndexError:
            error_str = f"The input has shape ({len(arr)}, {len(row)}) but the expected shape is (*, {len(ignorecolumns) + len(important_idxs)})."
            if len(arr) == num_attr and len(arr[0]) != num_attr:
                error_str += "\n\nNote: You may have passed an input directly to 'preprocess_and_clean_in_memory' or 'predict_in_memory' "
                error_str += "rather than as an element of a list. Make sure that even single instances "
                error_str += "are enclosed in a list. Example: predict_in_memory(0) is invalid but "
                error_str += "predict_in_memory([0]) is valid."
            raise PredictorError(error_str, 3)
        clean_arr[i] = [float(__convert(field)) for field in row_used_cols_only]
    return clean_arr


def __evaluate_tree(xs, split_vals, split_feats, right_children, logits):
    if xs is None:
        xs = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])

    current_node_per_row = np.zeros(xs.shape[0]).astype('int')
    values = np.empty(xs.shape[0])
    values.fill(np.nan)

    while np.isnan(values).any():

        row_idxs_at_leaf = np.argwhere(np.logical_and(right_children[current_node_per_row] == -1, np.isnan(values))).reshape(-1)
        row_idxs_at_branch = np.argwhere(right_children[current_node_per_row] != -1).reshape(-1)

        if row_idxs_at_leaf.shape[0] > 0:

            values[row_idxs_at_leaf] = logits[current_node_per_row[row_idxs_at_leaf]].reshape(-1)
            current_node_per_row[row_idxs_at_leaf] = -1

        if row_idxs_at_branch.shape[0] > 0:

            split_values_per_row = split_vals[current_node_per_row[row_idxs_at_branch]].astype('float64')
            split_features_per_row = split_feats[current_node_per_row[row_idxs_at_branch]].astype('int')
            feature_val_per_row = xs[row_idxs_at_branch, split_features_per_row].reshape(-1)

            branch_nodes = current_node_per_row[row_idxs_at_branch]
            current_node_per_row[row_idxs_at_branch] = np.where(feature_val_per_row < split_values_per_row,
                                                                right_children[branch_nodes].astype('int'),
                                                                (right_children[branch_nodes] + 1).astype('int'))

    return values


def __build_logit_func(n_trees, clss):

    def __logit_func(xs, serial, data_shape, pool=None):
        if serial:
            sum_of_leaf_values = np.zeros(xs.shape[0])
            for booster_index in range(clss, n_trees, n_classes):
                sum_of_leaf_values += __evaluate_tree(
                    xs, split_vals_dict[booster_index], split_feats_dict[booster_index],
                    right_children_dict[booster_index], logits_dict[booster_index])
        else:
            sum_of_leaf_values = np.sum(
                list(pool.starmap(__evaluate_tree,
                                  [(None, split_vals_dict[booster_index], split_feats_dict[booster_index],
                                    right_children_dict[booster_index], logits_dict[booster_index])
                                   for booster_index in range(clss, n_trees, n_classes)])), axis=0)
        return sum_of_leaf_values

    return __logit_func


def __init_worker(X, X_shape):
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape

def __classify(rows, return_probabilities=False, force_serial=False):
    if force_serial:
        serial = True
    else:
        serial = default_to_serial
    if isinstance(rows, list):
        rows = np.array(rows)

    logits = [__build_logit_func(24, clss) for clss in range(n_classes)]

    if serial:
        o = np.array([logits[class_index](rows, True, rows.shape) for class_index in range(n_classes)]).T
    else:
        shared_arr = multiprocessing.RawArray('d', rows.shape[0] * rows.shape[1])
        shared_arr_np = np.frombuffer(shared_arr, dtype=rows.dtype).reshape(rows.shape)
        np.copyto(shared_arr_np, rows)

        procs = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=procs, initializer=__init_worker, initargs=(shared_arr, rows.shape))
        o = np.array([logits[class_index](None, False, rows.shape, pool) for class_index in range(n_classes)]).T

    if return_probabilities:

        argument = o[:, 0] - o[:, 1]
        p0 = 1.0 / (1.0 + np.exp(-argument)).reshape(-1, 1)
        p1 = 1.0 - p0
        output = np.concatenate((p0, p1), axis=1)

    else:
        output = np.argmax(o, axis=1)
    return output



def __validate_kwargs(kwargs):
    for key in kwargs:

        if key not in ['return_probabilities', 'force_serial']:
            raise PredictorError(f'{key} is not a keyword argument for Brainome\'s {classifier_type} predictor. Please see the documentation.', 4)


def __validate_data(row_or_arr, validate, row_num=None):
    if validate:
        expected_columns = len(important_idxs) + len(ignore_idxs) + 1
    else:
        expected_columns = len(important_idxs) + len(ignore_idxs)

    input_is_array = isinstance(row_or_arr, np.ndarray)
    n_cols = row_or_arr.shape[1] if input_is_array else len(row_or_arr)

    if n_cols != expected_columns:

        if row_num is None:
            err_str = f"Your data contains {n_cols} columns but {expected_columns} are required."
        else:
            err_str = f"At row {row_num}, your data contains {n_cols} columns but {expected_columns} are required."

        if validate:
            err_str += " The predictor's validate() method works on data that has the same columns in the same order as were present in the training CSV."
            err_str += " This includes the target column and features that are not used by the model but existed in the training CSV."
            if n_cols == 1 + len(important_idxs):
                err_str += f" We suggest confirming that the {len(ignore_idxs)} unused features are present in the data."
            elif n_cols == len(important_idxs):
                err_str += f" We suggest confirming that the {len(ignore_idxs)} unused features are present in the data as well as the target column. "
            elif n_cols == len(important_idxs) + len(ignore_idxs):
                err_str += " We suggest confirming that the target column present in the data. "
            err_str += " To make predictions, see the predictor's predict() method."
        else:
            err_str += " The predictor's predict() method works on data that has the same feature columns in the same relative order as were present in the training CSV."
            err_str += " This DOES NOT include the target column but DOES include features that are not used by the model but existed in the training CSV."
            if n_cols == 1 + len(important_idxs):
                err_str += f" We suggest confirming that the {len(ignore_idxs)} unused features are present in the data and that the target column is not present."
            elif n_cols == len(important_idxs):
                err_str += f" We suggest confirming that the {len(ignore_idxs)} unused features are present in the data."
            elif n_cols == 1 + len(important_idxs) + len(ignore_idxs):
                err_str += " We suggest confirming that the target column is not present."
            err_str += " To receive a performance summary, instead of make predictions, see the predictor's validate() method."

        raise PredictorError(err_str, 5)

    else:

        if not input_is_array:
            return row_or_arr


def __write_predictions(arr, header, headerless, trim, outfile=None):
    predictions = predict(arr)

    if not headerless:
        if trim:
            header = ','.join([x for i, x in enumerate(header) if i in important_idxs] + ['Prediction'])
        else:
            header = ','.join(header.tolist() + ['Prediction'])
        if outfile is None:
            print(header)
        else:
            print(header, file=outfile)

    for row, prediction in zip(arr, predictions):
        if trim:
            row = ['"' + field + '"' if ',' in field else field for i, field in enumerate(row) if i in important_idxs]
        else:
            row = ['"' + field + '"' if ',' in field else field for field in row]
        row.append(prediction)
        if outfile is None:
            print(','.join(row))
        else:
            print(','.join(row), file=outfile)


def load_data(csvfile, headerless, validate):
    """
    Parameters
    ----------
    csvfile : str
        The path to the CSV file containing the data.

    headerless : bool
        True if the CSV does not contain a header.

    validate : bool
        True if the data should be loaded to be used by the predictor's validate() method.
        False if the data should be loaded to be used by the predictor's predict() method.

    Returns
    -------
    arr : np.ndarray
        The data (observations and labels) found in the CSV without any header.

    data : np.ndarray or NoneType
        None if validate is False, otherwise the observations (data without the labels) found in the CSV.

    labels : np.ndarray or NoneType
        None if the validate is False, otherwise the labels found in the CSV.

    header : np.ndarray or NoneType
        None if the CSV is headerless, otherwise the header.
    """

    with open(csvfile, 'r', encoding='utf-8') as csvinput:
        arr = np.array([__validate_data(row, validate, row_num=i) for i, row in enumerate(csv.reader(csvinput)) if row != []], dtype=str)
    if headerless:
        header = None
    else:
        header = arr[0]
        arr = arr[1:]
    if validate:
        labels = arr[:, target_column]
        feature_columns = [i for i in range(arr.shape[1]) if i != target_column]
        data = arr[:, feature_columns]
    else:
        data, labels = None, None

    if validate and ignorelabels != []:
        idxs_to_keep = np.argwhere(np.logical_not(np.isin(labels, ignorelabels))).reshape(-1)
        labels = labels[idxs_to_keep]
        data = data[idxs_to_keep]

    return arr, data, labels, header


def predict(arr, remap=True, **kwargs):
    """
    Parameters
    ----------
    arr : list[list]
        An array of inputs to be cleaned by 'preprocess_and_clean_in_memory'. This
        should contain all the features that were present in the training data,
        regardless of whether or not they are used by the model, with the same
        relative order as in the training data. There should be no target column.


    remap : bool
        If True and 'return_probs' is False, remaps the output to the original class
        label. If 'return_probs' is True this instead adds a header indicating which
        original class label each column of output corresponds to.

    **kwargs :
        return_probabilities : bool
            If true, return class membership probabilities instead of classifications.

    **kwargs :
        force_serial : bool
            If true, model inference is done in serial rather than in parallel. This is
            useful if calling "predict" repeatedly inside a for-loop.

    Returns
    -------
    output : np.ndarray

        A numpy array of
            1. Class predictions if 'return_probabilities' is False.
            2. Class probabilities if 'return_probabilities' is True.

    """
    if not isinstance(arr, np.ndarray) and not isinstance(arr, list):
        raise PredictorError(f'Data must be provided to \'predict\' and \'validate\' as a list or np.ndarray, but an input of type {type(arr).__name__} was found.', 6)
    if isinstance(arr, list):
        arr = np.array(arr, dtype=str)

    kwargs = kwargs or {}
    __validate_kwargs(kwargs)
    __validate_data(arr, False)
    remove_bad_chars = lambda x: str(x).replace('"', '').replace(',', '').replace('(', '').replace(')', '').replace("'", '')
    arr = [[remove_bad_chars(field) for field in row] for row in arr]
    arr = __preprocess_and_clean_in_memory(arr)

    output = __classify(arr, **kwargs)

    if remap:
        if kwargs.get('return_probabilities'):
            header = np.array([__get_key(i, mapping) for i in range(output.shape[1])], dtype=str).reshape(1, -1)
            output = np.concatenate((header, output), axis=0)
        else:
            output = np.array([__get_key(prediction, mapping) for prediction in output])

    return output


def validate(arr, labels):
    """
    Parameters
    ----------
    cleanarr : np.ndarray
        An array of float values that has undergone each pre-
        prediction step.

    Returns
    -------
    count : int
        A count of the number of instances in cleanarr.

    correct_count : int
        A count of the number of correctly classified instances in
        cleanarr.

    numeachclass : dict
        A dictionary mapping each class to its number of instances.

    outputs : np.ndarray
        The output of the predictor's '__classify' method on cleanarr.
    """
    predictions = predict(arr)
    correct_count = int(np.sum(predictions.reshape(-1) == labels.reshape(-1)))
    count = predictions.shape[0]
    
    class_0, class_1 = __get_key(0, mapping), __get_key(1, mapping)
    num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0
    num_TP = int(np.sum(np.logical_and(predictions.reshape(-1) == class_1, labels.reshape(-1) == class_1)))
    num_TN = int(np.sum(np.logical_and(predictions.reshape(-1) == class_0, labels.reshape(-1) == class_0)))
    num_FN = int(np.sum(np.logical_and(predictions.reshape(-1) == class_0, labels.reshape(-1) == class_1)))
    num_FP = int(np.sum(np.logical_and(predictions.reshape(-1) == class_1, labels.reshape(-1) == class_0)))
    num_class_0 = int(np.sum(labels.reshape(-1) == class_0))
    num_class_1 = int(np.sum(labels.reshape(-1) == class_1))
    return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, predictions


def __main():
    parser = argparse.ArgumentParser(description='Predictor trained on ' + str(TRAINFILE))
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    parser.add_argument('-json', action="store_true", default=False, help="report measurements as json")
    parser.add_argument('-trim', action="store_true", help="If true, the prediction will not output ignored columns.")
    args = parser.parse_args()
    faulthandler.enable()

    arr, data, labels, header = load_data(csvfile=args.csvfile, headerless=args.headerless, validate=args.validate)

    if not args.validate:
        __write_predictions(arr, header, args.headerless, args.trim)
    else:

        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds = validate(data, labels)

        classcounts = np.bincount(np.array([mapping[label] for label in labels], dtype='int32')).reshape(-1)
        class_balance = (classcounts[np.argwhere(classcounts > 0)] / arr.shape[0]).reshape(-1).tolist()
        best_guess = round(100.0 * np.max(class_balance), 2)
        H = float(-1.0 * sum([class_balance[i] * math.log(class_balance[i]) / math.log(2) for i in range(len(class_balance))]))
        modelacc = int(float(correct_count * 10000) / count) / 100.0
        mtrx, stats = __confusion_matrix(np.array(labels).reshape(-1), np.array(preds).reshape(-1), args.json)

        if args.json:
            json_dict = {'instance_count': count,
                         'classifier_type': classifier_type,
                         'classes': n_classes,
                         'number_correct': correct_count,
                         'accuracy': {
                             'best_guess': (best_guess/100),
                             'improvement': (modelacc - best_guess)/100,
                              'model_accuracy': (modelacc/100),
                         },
                         'model_capacity': model_cap,
                         'generalization_ratio': int(float(correct_count * 100) / model_cap) / 100.0 * H,
                         'model_efficiency': int(100 * (modelacc - best_guess) / model_cap) / 100.0,
                         'shannon_entropy_of_labels': H,
                         'class_balance': class_balance,
                         'confusion_matrix': mtrx.tolist(),
                         'multiclass_stats': stats}

            print(json.dumps(json_dict))
        else:
            pad = lambda s, length, pad_right: str(s) + ' ' * max(0, length - len(str(s))) if pad_right else ' ' * max(0, length - len(str(s))) + str(s)
            labels = np.array(list(mapping.keys())).reshape(-1, 1)
            max_class_name_len = max([len(clss) for clss in mapping.keys()] + [7])

            max_TP_len = max([len(str(stats[key]['TP'])) for key in stats.keys()] + [2])
            max_FP_len = max([len(str(stats[key]['FP'])) for key in stats.keys()] + [2])
            max_TN_len = max([len(str(stats[key]['TN'])) for key in stats.keys()] + [2])
            max_FN_len = max([len(str(stats[key]['FN'])) for key in stats.keys()] + [2])

            cmat_template_1 = "    {} | {}"
            cmat_template_2 = "    {} | " + " {} " * n_classes
            acc_by_class_template_1 = "    {} | {}  {}  {}  {}  {}  {}  {}  {}  {}  {}"

            acc_by_class_lengths = [max_class_name_len, max_TP_len, max_FP_len, max_TN_len, max_FN_len, 7, 7, 7, 7, 7, 7]
            acc_by_class_header_fields = ['target', 'TP', 'FP', 'TN', 'FN', 'TPR', 'TNR', 'PPV', 'NPV', 'F1', 'TS']
            print("Classifier Type:                    Random Forest")

            print(f"System Type:                        {n_classes}-way classifier\n")

            print("Accuracy:")
            print("    Best-guess accuracy:            {:.2f}%".format(best_guess))
            print("    Model accuracy:                 {:.2f}%".format(modelacc) + " (" + str(int(correct_count)) + "/" + str(count) + " correct)")
            print("    Improvement over best guess:    {:.2f}%".format(modelacc - best_guess) + " (of possible " + str(round(100 - best_guess, 2)) + "%)\n")

            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            print("Generalization ratio:               {:.2f}".format(int(float(correct_count * 100) / model_cap) / 100.0 * H) + " bits/bit")

            if report_cmat:
                max_cmat_entry_len = len(str(int(np.max(mtrx))))
                mtrx = np.concatenate((labels, mtrx.astype('str')), axis=1).astype('str')
                max_pred_len = (mtrx.shape[1] - 1) * max_cmat_entry_len + n_classes * 2 - 1
                print("\nConfusion Matrix:\n")
                print(cmat_template_1.format(pad("Actual", max_class_name_len, False), "Predicted"))
                print(cmat_template_1.format("-" * max_class_name_len, "-" * max(max_pred_len, 9)))
                for row in mtrx:
                    print(cmat_template_2.format(
                        *[pad(field, max_class_name_len if i == 0 else max_cmat_entry_len, False) for i, field in enumerate(row)]))

            print("\nAccuracy by Class:\n")
            print(acc_by_class_template_1.format(
                *[pad(header_field, length, False) for i, (header_field, length) in enumerate(zip(acc_by_class_header_fields, acc_by_class_lengths))]))
            print(acc_by_class_template_1.format(
                *["-" * length for length in acc_by_class_lengths]))

            pct_format_string = "{:8.2%}"      # width = 8, decimals = 2
            for raw_class in mapping.keys():
                class_stats = stats[int(mapping[raw_class])]
                TP, FP, TN, FN = class_stats.get('TP', None), class_stats.get('FP', None), class_stats.get('TN', None), class_stats.get('FN', None)
                TPR = pct_format_string.format(class_stats['TPR']) if class_stats['TPR'] is not None else 'N/A'
                TNR = pct_format_string.format(class_stats['TNR']) if class_stats['TNR'] is not None else 'N/A'
                PPV = pct_format_string.format(class_stats['PPV']) if class_stats['PPV'] is not None else 'N/A'
                NPV = pct_format_string.format(class_stats['NPV']) if class_stats['NPV'] is not None else 'N/A'
                F1 = pct_format_string.format(class_stats['F1']) if class_stats['F1'] is not None else 'N/A'
                TS = pct_format_string.format(class_stats['TS']) if class_stats['TS'] is not None else 'N/A'
                line_fields = [raw_class, TP, FP, TN, FN, TPR, TNR, PPV, NPV, F1, TS]
                print(acc_by_class_template_1.format(
                    *[pad(field, length, False) for i, (field, length) in enumerate(zip(line_fields, acc_by_class_lengths))]))


if __name__ == "__main__":
    try:
        __main()
    except PredictorError as e:
        print(e, file=sys.stderr)
        sys.exit(e.code)
    except Exception as e:
        print(f"An unknown exception of type {type(e).__name__} occurred.", file=sys.stderr)
        sys.exit(-1)
