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
# Invocation: brainome TRAIN_TEST_SPLITS/kdd_ipums_la_99-small-clean-train.csv -f RF -y -split 70 -modelonly -q -o btc-runs/RF/ipums_la_99_small.py -json btc-runs/RF/ipums_la_99_small.json
# Total compiler execution time: 0:00:06.93. Finished on: Feb-26-2022 18:29:56.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/RF/ipums_la_99_small.py
    Classifier Type:              Random Forest
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        93.57%
      Training accuracy:          94.09% (4076/4332 correct)
      Validation Accuracy:        93.86% (1744/1858 correct)
      Combined Model Accuracy:    94.02% (5820/6190 correct)


    Model Capacity (MEC):         49    bits
    Generalization Ratio:         28.60 bits/bit
    Percent of Data Memorized:    10.01%
    Resilience to Noise:          -1.92 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |  4042    12 
                   1 |   244    34 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |  1735     3 
                   1 |   111     9 

    Training Accuracy by Class:
         binaryClass |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
         ----------- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |  4042   244    34    12   99.70%   12.23%   94.31%   73.91%   96.93%   94.04%
                   1 |    34    12  4042   244   12.23%   99.70%   73.91%   94.31%   20.99%   11.72%

    Validation Accuracy by Class:
         binaryClass |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
         ----------- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |  1735   111     9     3   99.83%    7.50%   93.99%   75.00%   96.82%   93.83%
                   1 |     9     3  1735   111    7.50%   99.83%   75.00%   93.99%   13.64%    7.32%


    Attribute Ranking:
                                      Feature | Relative Importance
                                     labforce :   0.2336
                                      incwage :   0.1485
                                     yrlastwk :   0.0843
                                     empstatg :   0.0842
                                     hrswork2 :   0.0714
                                      poverty :   0.0610
                                        nfams :   0.0602
                                      famsize :   0.0469
                                      relateg :   0.0425
                                     occscore :   0.0369
                                      ftotinc :   0.0350
                                         rent :   0.0218
                                          age :   0.0192
                                           gq :   0.0187
                                          sei :   0.0114
                                       inctot :   0.0096
                                     wkswork2 :   0.0079
                                     ownershg :   0.0068
         

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
TRAINFILE = ['TRAIN_TEST_SPLITS/kdd_ipums_la_99-small-clean-train.csv']
mapping = {'0': 0, '1': 1}
ignorelabels = []
ignorecolumns = []
target = ''
target_column = 56
important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
ignore_idxs = []
classifier_type = 'RF'
num_attr = 56
n_classes = 2
model_cap = 49
logits_dict = {0: np.array([0.0, 0.0199814122, 0.0, 0.0, 0.0, 0.0196992476, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0120000001, 0.00999999978, 0.0115841581, -0.00666666683, 0.00588235306, -0.00838709623, 0.0164497029, 0.0185692981, -0.00999999978, 0.0133333337]), 1: np.array([0.0, -0.0199814122, 0.0, 0.0, 0.0, -0.0196992476, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0120000001, -0.00999999978, -0.0115841581, 0.00666666683, -0.00588235306, 0.00838709623, -0.0164497029, -0.0185692981, 0.00999999978, -0.0133333337]), 2: np.array([0.0, 0.0195900202, 0.0, 0.0, 0.0, 0.019356586, 0.0, 0.0, -0.00281168614, 0.0, 0.0, 0.0, 0.0, 0.00565263722, 0.0124788415, -0.00564981112, 0.0142449662, 0.0182619393, 0.00638421159, 0.0167894959, 0.00984377973]), 3: np.array([0.0, -0.0195900202, 0.0, 0.0, 0.0, -0.019356586, 0.0, 0.0, 0.00281168451, 0.0, 0.0, 0.0, 0.0, -0.00565263908, -0.0124788424, 0.00564981019, -0.0142449662, -0.0182619411, -0.00638421206, -0.0167894978, -0.00984378066]), 4: np.array([0.0, 0.0192211699, 0.0, 0.0, 0.0, 0.0189987328, 0.0, 0.0, -0.00277157151, 0.0, 0.0, 0.0, 0.0, 0.01113322, -0.00670942059, 0.00580252521, -0.00815157499, 0.0158003401, 0.0178761818, -0.00417560432, 0.0116336187]), 5: np.array([0.0, -0.0192211699, 0.0, 0.0, 0.0, -0.0189987309, 0.0, 0.0, 0.00277157221, 0.0, 0.0, 0.0, 0.0, -0.01113322, 0.00670942012, -0.00580252474, 0.00815157499, -0.0158003401, -0.0178761818, 0.00417560432, -0.0116336197]), 6: np.array([0.0, 0.0188730303, 0.0, 0.0, 0.0, 0.0186904576, 0.0, 0.0, -0.00273206038, 0.0, 0.0, 0.0, 0.0, 0.00533542922, 0.0120197572, -0.00548120309, 0.0140078673, 0.0176018979, 0.00601849006, 0.0161601994, 0.00934714638]), 7: np.array([0.0, -0.0188730303, 0.0, 0.0, 0.0, -0.0186904576, 0.0, 0.0, 0.00273206038, 0.0, 0.0, 0.0, 0.0, -0.00533542736, -0.0120197544, 0.00548120262, -0.0140078682, -0.0176018979, -0.00601849006, -0.0161601994, -0.00934714638]), 8: np.array([0.0, 0.0185439587, 0.0, 0.0, 0.0, 0.0183701087, 0.0, 0.0, -0.00269314065, 0.0, 0.0, 0.0, 0.0, 0.0107105579, -0.00674469117, 0.00572405662, -0.0079256976, 0.0172887612, 0.00592743745, 0.0158687327, 0.00917487498]), 9: np.array([0.0, -0.0185439605, 0.0, 0.0, 0.0, -0.0183701105, 0.0, 0.0, 0.00269314065, 0.0, 0.0, 0.0, 0.0, -0.0107105579, 0.00674469117, -0.00572405849, 0.0079256976, -0.0172887612, -0.00592743745, -0.0158687346, -0.00917487405]), 10: np.array([0.0, 0.0182324909, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00265480275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0109144812, -0.00598511007, 0.0181751121, 0.00299044186, 0.00345372525, -0.0103762364, 0.0056882184, 0.0131351836, 0.0149228368, 0.0169676617, -0.00451386999, 0.0111208064]), 11: np.array([0.0, -0.0182324927, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00265480205, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0109144812, 0.005985111, -0.0181751121, -0.0029904421, -0.00345372455, 0.0103762355, -0.0056882184, -0.0131351836, -0.0149228359, -0.0169676635, 0.00451386953, -0.0111208064]), 12: np.array([0.0, 0.0179373007, 0.0, 0.0, 0.0, 0.0177715719, 0.0, 0.0, -0.00261703646, 0.0, 0.0, 0.0, 0.0, 0.00825301465, 0.0136553757, 0.00560697308, -0.0077475463, 0.0167202838, 0.00558813848, 0.0152306147, 0.00789135508]), 13: np.array([0.0, -0.0179373007, 0.0, 0.0, 0.0, -0.0177715737, 0.0, 0.0, 0.00261703599, 0.0, 0.0, 0.0, 0.0, -0.00825301465, -0.0136553757, -0.00560697168, 0.0077475463, -0.0167202819, -0.00558813848, -0.0152306147, -0.00789135415]), 14: np.array([0.0, 0.0176571961, 0.0, 0.0, 0.0, 0.0175547693, 0.0, 0.0, -0.00257983105, 0.0, 0.0, 0.0, 0.0, 0.000444717181, 0.013857861, 0.00842321478, 0.0136309946, 0.0164488424, 0.00550485728, 0.0150649846, 0.00855169352]), 15: np.array([0.0, -0.0176571961, 0.0, 0.0, 0.0, -0.0175547674, 0.0, 0.0, 0.00257983105, 0.0, 0.0, 0.0, 0.0, -0.000444716163, -0.013857861, -0.00842321478, -0.0136309946, -0.0164488442, -0.00550485728, -0.0150649855, -0.00855169445]), 16: np.array([0.0, 0.0173910968, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00254317792, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0103583615, -0.0061576548, 0.0174226779, 0.00253403326, 0.00308536319, -0.0102208173, 0.00525026955, 0.0125223249, 0.014149569, 0.0161794163, -0.00483084377, 0.0106580956]), 17: np.array([0.0, -0.0173910968, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00254317815, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0103583615, 0.00615765434, -0.0174226798, -0.0025340342, -0.00308536435, 0.0102208173, -0.00525027094, -0.0125223249, -0.0141495699, -0.0161794163, 0.00483084377, -0.0106580975])}
right_children_dict = {0: np.array([1, -1, 3, 5, 7, -1, 9, 11, 13, 15, 17, 19, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 1: np.array([1, -1, 3, 5, 7, -1, 9, 11, 13, 15, 17, 19, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 2: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 3: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 4: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 5: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 6: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 7: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 8: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 9: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 10: np.array([1, -1, 3, 5, 7, 9, 11, 13, -1, 15, 17, 19, 21, 23, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 11: np.array([1, -1, 3, 5, 7, 9, 11, 13, -1, 15, 17, 19, 21, 23, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 12: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 13: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 14: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 15: np.array([1, -1, 3, 5, 7, -1, 9, 11, -1, 13, 15, 17, 19, -1, -1, -1, -1, -1, -1, -1, -1]), 16: np.array([1, -1, 3, 5, 7, 9, 11, 13, -1, 15, 17, 19, 21, 23, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 17: np.array([1, -1, 3, 5, 7, 9, 11, 13, -1, 15, 17, 19, 21, 23, 25, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])}
split_feats_dict = {0: np.array([37, 0, 46, 43, 1, 0, 8, 41, 39, 7, 6, 38, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1: np.array([37, 0, 46, 43, 1, 0, 8, 41, 39, 7, 6, 38, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2: np.array([37, 0, 46, 36, 1, 0, 8, 4, 0, 38, 38, 45, 28, 0, 0, 0, 0, 0, 0, 0, 0]), 3: np.array([37, 0, 46, 36, 1, 0, 8, 4, 0, 38, 38, 45, 28, 0, 0, 0, 0, 0, 0, 0, 0]), 4: np.array([37, 0, 46, 43, 1, 0, 8, 41, 0, 7, 6, 38, 20, 0, 0, 0, 0, 0, 0, 0, 0]), 5: np.array([37, 0, 46, 43, 1, 0, 8, 41, 0, 7, 6, 38, 20, 0, 0, 0, 0, 0, 0, 0, 0]), 6: np.array([37, 0, 46, 36, 1, 0, 8, 4, 0, 38, 38, 45, 28, 0, 0, 0, 0, 0, 0, 0, 0]), 7: np.array([37, 0, 46, 36, 1, 0, 8, 4, 0, 38, 38, 45, 28, 0, 0, 0, 0, 0, 0, 0, 0]), 8: np.array([37, 0, 46, 43, 1, 0, 8, 4, 0, 7, 6, 45, 28, 0, 0, 0, 0, 0, 0, 0, 0]), 9: np.array([37, 0, 46, 43, 1, 0, 8, 4, 0, 7, 6, 45, 28, 0, 0, 0, 0, 0, 0, 0, 0]), 10: np.array([37, 0, 46, 42, 1, 42, 20, 41, 0, 7, 45, 8, 38, 38, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 11: np.array([37, 0, 46, 42, 1, 42, 20, 41, 0, 7, 45, 8, 38, 38, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 12: np.array([37, 0, 46, 43, 1, 0, 8, 4, 0, 52, 6, 45, 28, 0, 0, 0, 0, 0, 0, 0, 0]), 13: np.array([37, 0, 46, 43, 1, 0, 8, 4, 0, 52, 6, 45, 28, 0, 0, 0, 0, 0, 0, 0, 0]), 14: np.array([37, 0, 46, 36, 1, 0, 38, 4, 0, 52, 27, 45, 28, 0, 0, 0, 0, 0, 0, 0, 0]), 15: np.array([37, 0, 46, 36, 1, 0, 38, 4, 0, 52, 27, 45, 28, 0, 0, 0, 0, 0, 0, 0, 0]), 16: np.array([37, 0, 46, 42, 1, 42, 20, 41, 0, 7, 45, 8, 38, 38, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 17: np.array([37, 0, 46, 42, 1, 42, 20, 41, 0, 7, 45, 8, 38, 38, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
split_vals_dict = {0: np.array([3912424960.0, 0.0, 13097.0, 1664895230.0, 1632849790.0, 0.0, 3.5, 2082328060.0, 20.5, 282669.5, 524.5, 25.5, 35.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1: np.array([3912424960.0, 0.0, 13097.0, 1664895230.0, 1632849790.0, 0.0, 3.5, 2082328060.0, 20.5, 282669.5, 524.5, 25.5, 35.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2: np.array([3912424960.0, 0.0, 13097.0, 2647492100.0, 1632849790.0, 0.0, 3.5, 1247051780.0, 0.0, 18.5, 29.0, 216296.0, 50.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3: np.array([3912424960.0, 0.0, 13097.0, 2647492100.0, 1632849790.0, 0.0, 3.5, 1247051780.0, 0.0, 18.5, 29.0, 216296.0, 50.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 4: np.array([3912424960.0, 0.0, 13097.0, 1664895230.0, 1632849790.0, 0.0, 3.5, 2082328060.0, 0.0, 282669.5, 524.5, 25.5, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 5: np.array([3912424960.0, 0.0, 13097.0, 1664895230.0, 1632849790.0, 0.0, 3.5, 2082328060.0, 0.0, 282669.5, 524.5, 25.5, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 6: np.array([3912424960.0, 0.0, 13097.0, 2647492100.0, 1632849790.0, 0.0, 3.5, 1247051780.0, 0.0, 18.5, 29.0, 216296.0, 50.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 7: np.array([3912424960.0, 0.0, 13097.0, 2647492100.0, 1632849790.0, 0.0, 3.5, 1247051780.0, 0.0, 18.5, 29.0, 216296.0, 50.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 8: np.array([3912424960.0, 0.0, 13097.0, 1664895230.0, 1632849790.0, 0.0, 3.5, 1247051780.0, 0.0, 282669.5, 524.5, 216296.0, 50.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 9: np.array([3912424960.0, 0.0, 13097.0, 1664895230.0, 1632849790.0, 0.0, 3.5, 1247051780.0, 0.0, 282669.5, 524.5, 216296.0, 50.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 10: np.array([3912424960.0, 0.0, 13097.0, 1892011260.0, 1632849790.0, 1275573760.0, 1.5, 2082328060.0, 0.0, 177115.0, 46806.0, 3.5, 22.5, 25.5, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 11: np.array([3912424960.0, 0.0, 13097.0, 1892011260.0, 1632849790.0, 1275573760.0, 1.5, 2082328060.0, 0.0, 177115.0, 46806.0, 3.5, 22.5, 25.5, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 12: np.array([3912424960.0, 0.0, 13097.0, 1664895230.0, 1632849790.0, 0.0, 3.5, 1247051780.0, 0.0, 316.5, 524.5, 216296.0, 52.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 13: np.array([3912424960.0, 0.0, 13097.0, 1664895230.0, 1632849790.0, 0.0, 3.5, 1247051780.0, 0.0, 316.5, 524.5, 216296.0, 52.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 14: np.array([3912424960.0, 0.0, 13097.0, 2647492100.0, 1632849790.0, 0.0, 18.5, 1247051780.0, 0.0, 323.5, 2401795330.0, 216296.0, 50.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 15: np.array([3912424960.0, 0.0, 13097.0, 2647492100.0, 1632849790.0, 0.0, 18.5, 1247051780.0, 0.0, 323.5, 2401795330.0, 216296.0, 50.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 16: np.array([3912424960.0, 0.0, 13097.0, 1892011260.0, 1632849790.0, 1275573760.0, 1.5, 2082328060.0, 0.0, 177115.0, 46806.0, 3.5, 22.5, 25.5, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 17: np.array([3912424960.0, 0.0, 13097.0, 1892011260.0, 1632849790.0, 1275573760.0, 1.5, 2082328060.0, 0.0, 177115.0, 46806.0, 3.5, 22.5, 25.5, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}


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

    logits = [__build_logit_func(18, clss) for clss in range(n_classes)]

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
