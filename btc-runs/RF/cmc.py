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
# Invocation: brainome TRAIN_TEST_SPLITS/cmc-clean-train.csv -f RF -y -split 70 -modelonly -q -o btc-runs/RF/cmc.py -json btc-runs/RF/cmc.json
# Total compiler execution time: 0:00:06.09. Finished on: Feb-26-2022 18:28:47.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/RF/cmc.py
    Classifier Type:              Random Forest
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        57.32%
      Training accuracy:          81.27% (586/721 correct)
      Validation Accuracy:        73.54% (228/310 correct)
      Combined Model Accuracy:    78.95% (814/1031 correct)


    Model Capacity (MEC):         75    bits
    Generalization Ratio:          7.69 bits/bit
    Percent of Data Memorized:    26.16%
    Resilience to Noise:          -0.89 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   1 |   380    33 
                   0 |   102   206 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   1 |   152    26 
                   0 |    56    76 

    Training Accuracy by Class:
         binaryClass |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
         ----------- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   1 |   380   102   206    33   92.01%   66.88%   78.84%   86.19%   84.92%   73.79%
                   0 |   206    33   380   102   66.88%   92.01%   86.19%   78.84%   75.32%   60.41%

    Validation Accuracy by Class:
         binaryClass |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
         ----------- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   1 |   152    56    76    26   85.39%   57.58%   73.08%   74.51%   78.76%   64.96%
                   0 |    76    26   152    56   57.58%   85.39%   74.51%   73.08%   64.96%   48.10%


    Attribute Ranking:
                                      Feature | Relative Importance
                 Number_of_children_ever_born :   0.1875
                               Media_exposure :   0.1704
                              Wifes_education :   0.1448
                                    Wifes_age :   0.1234
                               Wifes_religion :   0.1080
                            Wifes_now_working :   0.0784
                     Standard_of_living_index :   0.0722
                          Husbands_occupation :   0.0613
                           Husbands_education :   0.0540
         

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
TRAINFILE = ['TRAIN_TEST_SPLITS/cmc-clean-train.csv']
mapping = {'1': 0, '0': 1}
ignorelabels = []
ignorecolumns = []
target = ''
target_column = 9
important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ignore_idxs = []
classifier_type = 'RF'
num_attr = 9
n_classes = 2
model_cap = 75
logits_dict = {0: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.239150777, 0.0, 0.0, 0.0, 0.0, 0.0, -0.27093333, 0.0, 0.0, 0.0, 0.0, -0.203200012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.152400002, -0.152400002, -0.0914400071, 0.152400002, 0.254000008, 0.107576475, 0.101600006, -0.0, -0.193963632, -0.0609600022, 0.00983225834, 0.12337143, -0.0247135125, -0.193963632, -0.0609600022, -0.274320006, -0.101600006, 0.130628571, -0.0, 0.169333339, 0.182880014, 0.0544285737, -0.0, -0.152400002]), 1: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.239150777, 0.0, 0.0, 0.0, 0.0, 0.0, 0.27093333, 0.0, 0.0, 0.0, 0.0, 0.203200012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.152400002, 0.152400002, 0.0914400071, -0.152400002, -0.254000008, -0.107576475, -0.101600006, -0.0, 0.193963632, 0.0609600022, -0.00983225834, -0.12337143, 0.0247135125, 0.193963632, 0.0609600022, 0.274320006, 0.101600006, -0.130628571, -0.0, -0.169333339, -0.182880014, -0.0544285737, -0.0, 0.152400002]), 2: np.array([0.0, 0.0, 0.0, 0.0, -0.237682968, 0.0, 0.0, -0.168364853, 0.0579983443, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00682451762, -0.155688375, 0.11146576, -0.0464097522, -0.0479732379, 0.155613884, -0.165543362, -0.0650392473, 0.212525442, 0.106728449, -0.030778164, -0.203739673, -0.151675344, 0.159697995, 0.155492291, -0.112871461]), 3: np.array([0.0, 0.0, 0.0, 0.0, 0.237682983, 0.0, 0.0, 0.168364868, -0.0579983331, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0068245139, 0.15568839, -0.11146576, 0.0464097522, 0.0479732454, -0.155613884, 0.165543362, 0.0650392473, -0.212525457, -0.106728449, 0.0307781771, 0.203739688, 0.151675344, -0.159697995, -0.155492291, 0.112871461]), 4: np.array([0.0, 0.0, 0.0, 0.0, -0.201162785, 0.0, 0.0, 0.0566544309, -0.149737909, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0938764289, 0.0569850653, -0.142370075, -0.176917136, 0.0436309762, -0.0723432675, -0.192899004, -0.137649834, 0.0719870701, 0.046142865, 0.16140987, -0.00133342191, -0.19624041, 0.123058662, 0.054673288]), 5: np.array([0.0, 0.0, 0.0, 0.0, 0.201162755, 0.0, 0.0, -0.0566544496, 0.149737909, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0938764289, -0.0569850579, 0.14237009, 0.176917151, -0.0436309949, 0.0723432675, 0.192899004, 0.137649834, -0.0719870776, -0.0461428687, -0.161409885, 0.00133342703, 0.19624041, -0.123058684, -0.0546732992]), 6: np.array([0.0, 0.0, 0.0, 0.0, -0.184132144, 0.0, 0.0, 0.0416444056, -0.123579912, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.132385656, 0.0, 0.0, 0.0, -0.114591688, 0.00407102751, -0.0391746685, 0.0737540126, 0.0388026834, -0.0585096702, -0.016071571, -0.162012056, -0.151705518, 0.125057578, 0.0644132122, 0.145226926, -0.147714138, 0.0333084725]), 7: np.array([0.0, 0.0, 0.0, 0.0, 0.184132144, 0.0, 0.0, -0.0416444242, 0.123579897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.132385656, 0.0, 0.0, 0.0, 0.114591688, -0.00407102192, 0.0391746685, -0.07375402, -0.0388026722, 0.0585096702, 0.0160715748, 0.162012056, 0.151705518, -0.125057593, -0.0644132048, -0.145226926, 0.147714138, -0.0333084501]), 8: np.array([0.0, 0.0, 0.0, 0.0, -0.166748106, 0.0, 0.0, -0.113174126, 0.0580470897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.156438693, 0.0140633108, -0.20680429, 0.0, -0.158718824, 0.0362915844, -0.180504605, -0.0868504569, 0.0891148746, -0.0396728776, 0.0696630105, -0.000579950516, 0.000201066461, -0.130483866]), 9: np.array([0.0, 0.0, 0.0, 0.0, 0.166748106, 0.0, 0.0, 0.113174118, -0.0580470897, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.156438693, -0.0140633015, 0.20680429, 0.0, 0.158718824, -0.0362915844, 0.180504635, 0.0868504569, -0.0891148746, 0.0396728925, -0.0696630105, 0.000579970074, -0.00020107429, 0.130483866]), 10: np.array([0.0, 0.0, 0.0, 0.0, -0.158361897, 0.0, 0.0, 0.0480799675, -0.099258475, 0.0, 0.0, 0.0, -0.107542902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.184273496, 0.0229305662, -0.0561970174, -0.0750930309, 0.104259297, -0.182791263, -0.0457575507, -0.0566844009, 0.023488652, 0.0789680108, 0.0249129105]), 11: np.array([0.0, 0.0, 0.0, 0.0, 0.158361897, 0.0, 0.0, -0.04807996, 0.0992584974, 0.0, 0.0, 0.0, 0.107542902, 0.0, 0.0, 0.0, 0.0, 0.0, -0.184273511, -0.0229305606, 0.0561970212, 0.0750930309, -0.104259282, 0.182791248, 0.045757547, 0.0566844009, -0.0234886445, -0.0789680108, -0.0249129143]), 12: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.113885745, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0674686879, 0.0, 0.0, -0.168444395, 0.0, 0.0, 0.0544484332, -0.160787269, 0.0, 0.0, 0.0, 0.0, -0.0156841446, -0.14102152, -0.0585468411, 0.0531422757, -0.138369292, -0.0513195954, -0.131597325, 0.0071646357, 0.127132624, -0.0885082409, -0.0160156656, 0.0668160543, 0.0991349146, -0.072576046, 0.0299947169, -0.164691806, 0.0613550134, -0.0995832458, -0.0314383209, 0.0689275861]), 13: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.113885768, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0674686879, 0.0, 0.0, 0.168444395, 0.0, 0.0, -0.0544484183, 0.160787269, 0.0, 0.0, 0.0, 0.0, 0.0156841315, 0.141021505, 0.0585468411, -0.0531422757, 0.138369292, 0.0513196029, 0.13159731, -0.00716463756, -0.127132624, 0.0885082409, 0.0160156563, -0.0668160543, -0.0991348997, 0.072576046, -0.0299947225, 0.164691806, -0.0613550097, 0.0995832458, 0.0314383022, -0.068927601])}
right_children_dict = {0: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, -1, 21, 23, 25, 27, 29, -1, 31, 33, 35, 37, -1, 39, 41, 43, 45, 47, 49, 51, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 1: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, -1, 21, 23, 25, 27, 29, -1, 31, 33, 35, 37, -1, 39, 41, 43, 45, 47, 49, 51, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 2: np.array([1, 3, 5, 7, -1, 9, 11, -1, -1, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 3: np.array([1, 3, 5, 7, -1, 9, 11, -1, -1, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 4: np.array([1, 3, 5, 7, -1, 9, 11, -1, -1, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 5: np.array([1, 3, 5, 7, -1, 9, 11, -1, -1, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 6: np.array([1, 3, 5, 7, -1, 9, 11, -1, -1, 13, 15, 17, 19, 21, 23, 25, 27, -1, 29, 31, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 7: np.array([1, 3, 5, 7, -1, 9, 11, -1, -1, 13, 15, 17, 19, 21, 23, 25, 27, -1, 29, 31, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 8: np.array([1, 3, 5, 7, -1, 9, 11, -1, -1, 13, 15, 17, 19, 21, 23, 25, 27, -1, -1, -1, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 9: np.array([1, 3, 5, 7, -1, 9, 11, -1, -1, 13, 15, 17, 19, 21, 23, 25, 27, -1, -1, -1, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 10: np.array([1, 3, 5, 7, -1, 9, 11, -1, -1, 13, 15, 17, -1, 19, 21, 23, 25, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 11: np.array([1, 3, 5, 7, -1, 9, 11, -1, -1, 13, 15, 17, -1, 19, 21, 23, 25, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 12: np.array([1, 3, 5, 7, 9, 11, 13, -1, 15, 17, 19, 21, 23, 25, 27, 29, 31, -1, 33, 35, -1, 37, 39, -1, -1, 41, 43, 45, 47, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 13: np.array([1, 3, 5, 7, 9, 11, 13, -1, 15, 17, 19, 21, 23, 25, 27, 29, 31, -1, 33, 35, -1, 37, 39, -1, -1, 41, 43, 45, 47, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])}
split_feats_dict = {0: np.array([3, 0, 1, 3, 0, 0, 0, 2, 1, 4, 0, 0, 8, 1, 8, 0, 0, 6, 6, 0, 0, 0, 3, 0, 2, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1: np.array([3, 0, 1, 3, 0, 0, 0, 2, 1, 4, 0, 0, 8, 1, 8, 0, 0, 6, 6, 0, 0, 0, 3, 0, 2, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2: np.array([3, 2, 1, 6, 0, 0, 3, 0, 0, 3, 4, 0, 7, 0, 3, 3, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 3: np.array([3, 2, 1, 6, 0, 0, 3, 0, 0, 3, 4, 0, 7, 0, 3, 3, 7, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4: np.array([3, 2, 1, 0, 0, 0, 3, 0, 0, 3, 1, 0, 8, 2, 0, 3, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 5: np.array([3, 2, 1, 0, 0, 0, 3, 0, 0, 3, 1, 0, 8, 2, 0, 3, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6: np.array([3, 0, 1, 1, 0, 0, 7, 0, 0, 3, 8, 0, 0, 6, 0, 6, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7: np.array([3, 0, 1, 1, 0, 0, 7, 0, 0, 3, 8, 0, 0, 6, 0, 6, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 8: np.array([3, 2, 8, 6, 0, 3, 0, 0, 0, 0, 7, 1, 3, 1, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9: np.array([3, 2, 8, 6, 0, 3, 0, 0, 0, 0, 7, 1, 3, 1, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10: np.array([3, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 3, 0, 7, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 11: np.array([3, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 3, 0, 7, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 12: np.array([3, 0, 7, 6, 0, 0, 0, 0, 6, 2, 4, 3, 6, 2, 0, 0, 7, 0, 7, 0, 0, 5, 0, 0, 0, 7, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 13: np.array([3, 0, 7, 6, 0, 0, 0, 0, 6, 2, 4, 3, 6, 2, 0, 0, 7, 0, 7, 0, 0, 5, 0, 0, 0, 7, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
split_vals_dict = {0: np.array([1.5, 24.5, 2.5, 0.5, 27.5, 36.5, 24.5, 2.5, 2.5, 0.5, 0.0, 20.5, 0.5, 3.5, 0.5, 21.5, 0.0, 3.5, 1.5, 26.5, 26.5, 0.0, 2.5, 46.5, 1.5, 2.5, 1.5, 42.5, 31.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1: np.array([1.5, 24.5, 2.5, 0.5, 27.5, 36.5, 24.5, 2.5, 2.5, 0.5, 0.0, 20.5, 0.5, 3.5, 0.5, 21.5, 0.0, 3.5, 1.5, 26.5, 26.5, 0.0, 2.5, 46.5, 1.5, 2.5, 1.5, 42.5, 31.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2: np.array([0.5, 2.5, 3.5, 2.5, 0.0, 37.5, 1.5, 0.0, 0.0, 2.5, 0.5, 25.5, 2.5, 30.5, 7.5, 3.5, 3.5, 1.5, 37.5, 36.5, 48.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3: np.array([0.5, 2.5, 3.5, 2.5, 0.0, 37.5, 1.5, 0.0, 0.0, 2.5, 0.5, 25.5, 2.5, 30.5, 7.5, 3.5, 3.5, 1.5, 37.5, 36.5, 48.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 4: np.array([0.5, 2.5, 2.5, 22.5, 0.0, 37.5, 2.5, 0.0, 0.0, 1.5, 1.5, 35.5, 0.5, 2.5, 20.5, 4.5, 3.5, 33.5, 0.5, 39.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 5: np.array([0.5, 2.5, 2.5, 22.5, 0.0, 37.5, 2.5, 0.0, 0.0, 1.5, 1.5, 35.5, 0.5, 2.5, 20.5, 4.5, 3.5, 33.5, 0.5, 39.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 6: np.array([0.5, 21.5, 3.5, 2.5, 0.0, 35.5, 2.5, 0.0, 0.0, 1.5, 0.5, 23.5, 42.5, 2.5, 24.5, 1.5, 36.5, 0.0, 36.5, 34.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 7: np.array([0.5, 21.5, 3.5, 2.5, 0.0, 35.5, 2.5, 0.0, 0.0, 1.5, 0.5, 23.5, 42.5, 2.5, 24.5, 1.5, 36.5, 0.0, 36.5, 34.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 8: np.array([0.5, 2.5, 0.5, 2.5, 0.0, 1.5, 24.5, 0.0, 0.0, 27.5, 2.5, 1.5, 2.5, 1.5, 0.5, 1.5, 42.5, 0.0, 0.0, 0.0, 37.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 9: np.array([0.5, 2.5, 0.5, 2.5, 0.0, 1.5, 24.5, 0.0, 0.0, 27.5, 2.5, 1.5, 2.5, 1.5, 0.5, 1.5, 42.5, 0.0, 0.0, 0.0, 37.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 10: np.array([0.5, 21.5, 3.5, 3.5, 0.0, 35.5, 48.5, 0.0, 0.0, 32.5, 2.5, 5.5, 0.0, 3.5, 1.5, 38.5, 3.5, 25.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 11: np.array([0.5, 21.5, 3.5, 3.5, 0.0, 35.5, 48.5, 0.0, 0.0, 32.5, 2.5, 5.5, 0.0, 3.5, 1.5, 38.5, 3.5, 25.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 12: np.array([1.5, 24.5, 2.5, 1.5, 31.5, 38.5, 23.5, 0.0, 2.5, 2.5, 0.5, 2.5, 1.5, 3.5, 42.5, 22.5, 1.5, 0.0, 2.5, 35.5, 0.0, 0.5, 32.5, 0.0, 0.0, 3.5, 21.5, 7.5, 6.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 13: np.array([1.5, 24.5, 2.5, 1.5, 31.5, 38.5, 23.5, 0.0, 2.5, 2.5, 0.5, 2.5, 1.5, 3.5, 42.5, 22.5, 1.5, 0.0, 2.5, 35.5, 0.0, 0.5, 32.5, 0.0, 0.0, 3.5, 21.5, 7.5, 6.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}


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

    logits = [__build_logit_func(14, clss) for clss in range(n_classes)]

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
