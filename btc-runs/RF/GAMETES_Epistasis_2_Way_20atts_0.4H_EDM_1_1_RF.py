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
# Invocation: brainome TRAIN_TEST_SPLITS/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1-clean-train.csv -f RF -y -split 70 -modelonly -q -o btc-runs/RF/GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1_RF.py -json btc-runs/RF/GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1_RF.json
# Total compiler execution time: 0:00:06.15. Finished on: Feb-26-2022 18:29:25.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/RF/GAMETES_Epistasis_2_Way_20atts_0.4H_EDM_1_1_RF.py
    Classifier Type:              Random Forest
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        50.00%
      Training accuracy:          59.69% (468/784 correct)
      Validation Accuracy:        53.27% (179/336 correct)
      Combined Model Accuracy:    57.76% (647/1120 correct)


    Model Capacity (MEC):         71    bits
    Generalization Ratio:          6.59 bits/bit
    Percent of Data Memorized:    30.35%
    Resilience to Noise:          -0.82 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   1 |   248   144 
                   0 |   172   220 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   1 |    95    73 
                   0 |    84    84 

    Training Accuracy by Class:
               class |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   1 |   248   172   220   144   63.27%   56.12%   59.05%   60.44%   61.08%   43.97%
                   0 |   220   144   248   172   56.12%   63.27%   60.44%   59.05%   58.20%   41.04%

    Validation Accuracy by Class:
               class |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   1 |    95    84    84    73   56.55%   50.00%   53.07%   53.50%   54.76%   37.70%
                   0 |    84    73    95    84   50.00%   56.55%   53.50%   53.07%   51.69%   34.85%


    Attribute Ranking:
                                      Feature | Relative Importance
                                           N8 :   0.1009
                                           N2 :   0.0977
                                           P1 :   0.0732
                                           N5 :   0.0693
                                           N0 :   0.0673
                                          N11 :   0.0664
                                           P2 :   0.0521
                                           N6 :   0.0484
                                           N9 :   0.0462
                                          N10 :   0.0443
                                          N17 :   0.0402
                                           N4 :   0.0402
                                          N16 :   0.0367
                                           N7 :   0.0362
                                          N13 :   0.0358
                                           N3 :   0.0335
                                           N1 :   0.0307
                                          N14 :   0.0284
                                          N12 :   0.0268
                                          N15 :   0.0258
         

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
TRAINFILE = ['TRAIN_TEST_SPLITS/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1-clean-train.csv']
mapping = {'1': 0, '0': 1}
ignorelabels = []
ignorecolumns = []
target = ''
target_column = 20
important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
ignore_idxs = []
classifier_type = 'RF'
num_attr = 20
n_classes = 2
model_cap = 71
logits_dict = {0: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0492461547, -0.0291000009, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0349200033, 0.0, -0.0, -0.00159016391, -0.0125102811, 0.000158583105, 0.0127756102, -0.0388000011, -0.0, -0.0349200033, 0.0349200033, -0.0, 0.0291000009, 0.0388000011, 0.0116400002]), 1: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0492461547, 0.0291000009, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0349200033, 0.0, -0.0, 0.00159016391, 0.0125102811, -0.000158583105, -0.0127756102, 0.0388000011, -0.0, 0.0349200033, -0.0349200033, -0.0, -0.0291000009, -0.0388000011, -0.0116400002]), 2: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.045364935, -0.000390648755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0188409332, -0.033725746, 0.0, -0.0, -0.00149862829, -0.0117976414, 0.000149405882, 0.0120521616, -0.0369654745, -0.0, -0.033725746, 0.033725746, 0.0370930359, 0.0109215043]), 3: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.045364935, 0.00039064759, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0188409351, 0.033725746, 0.0, -0.0, 0.0014986333, 0.0117976367, -0.000149404164, -0.0120521626, 0.0369654708, -0.0, 0.033725746, -0.033725746, -0.0370930396, -0.0109215053]), 4: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0434426442, -0.000375497446, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0181211904, -0.032618586, 0.0, -0.0, -0.00925688259, 0.000549380027, 0.00014075308, 0.0113730021, -0.0353006124, -0.0, -0.032618586, 0.032618586, 0.0354168564, 0.0105439294]), 5: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0434426442, 0.000375496282, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0181211941, 0.032618586, 0.0, -0.0, 0.00925688259, -0.000549380609, -0.00014075308, -0.011373003, 0.0353006087, -0.0, 0.032618586, -0.032618586, -0.0354168527, -0.0105439257]), 6: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0417282768, -0.000360933307, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0174367428, -0.0315889269, 0.0, -0.0, -0.00114781875, -0.0108063444, 0.000132603993, 0.0107347257, -0.0337797441, -0.0, -0.0315889269, 0.0315889269, 0.0339929052, 0.00991069898]), 7: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0417282768, 0.000360933307, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0174367428, 0.0315889269, 0.0, -0.0, 0.0011478205, 0.0108063472, -0.00013261089, -0.0107347211, 0.0337797441, -0.0, 0.0315889269, -0.0315889269, -0.0339929052, -0.00991069991]), 8: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0401886478, -0.000346937828, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0167849306, -0.0306284446, 0.0, -0.0, -0.00843505189, 0.000753201544, 0.000124931015, 0.0101342909, -0.00943984371, -0.0389210805, -0.0306284446, 0.0306284446, 0.0325795338, 0.00956925005]), 9: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0401886478, 0.000346935529, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0167849306, 0.0306284446, 0.0, -0.0, 0.00843505003, -0.000753205153, -0.000124931015, -0.0101342965, 0.00943984557, 0.0389210805, 0.0306284446, -0.0306284446, -0.0325795338, -0.00956925005]), 10: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.038797047, -0.000333480828, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0161633845, -0.0297299139, 0.0, -0.0, -0.000847708783, -0.0099103488, 0.000117699645, 0.00956901349, -0.00904586911, -0.0375386216, -0.0297299139, 0.0297299139, 0.0313716009, 0.00898913201]), 11: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0387970544, 0.000333479664, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0161633864, 0.0297299139, 0.0, -0.0, 0.000847703661, 0.00991035253, -0.000117704825, -0.00956900697, 0.00904586911, 0.0375386216, 0.0297299139, -0.0297299139, -0.0313716009, -0.00898913108]), 12: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0375317894, -0.000320546096, 0.0, 0.0, 0.0, 0.0128831035, 0.0, 0.0, 0.0, 0.0, -0.0414808393, -0.011121762, -0.00369747379, 0.00216720696, -0.0256576408, -1.33740363e-09, 0.0155699821, -0.0288870316, 0.0269629396, -0.0]), 13: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0375317857, 0.000320548424, 0.0, 0.0, 0.0, -0.0128831035, 0.0, 0.0, 0.0, 0.0, 0.0414808355, 0.0111217601, 0.00369747193, -0.00216721138, 0.0256576426, -1.33740363e-09, -0.0155699858, 0.0288870279, -0.0269629396, -0.0]), 14: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0363750346, -0.000308115094, 0.0, 0.0, 0.0, 0.0126355877, 0.0, 0.0, 0.0, 0.0, -0.0401696973, -0.0104215574, -0.00348461978, 0.00204177364, -0.0245792829, -1.33740363e-09, 0.0150028253, -0.0280942731, 0.0349321067, 0.0116855595]), 15: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0363750383, 0.000308115094, 0.0, 0.0, 0.0, -0.0126355877, 0.0, 0.0, 0.0, 0.0, 0.0401696973, 0.0104215601, 0.00348461908, -0.00204177247, 0.0245792791, -1.33740363e-09, -0.0150028253, 0.0280942712, -0.034932103, -0.0116855558]), 16: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0353120416, -0.000296165817, 0.0, 0.0, 0.0, 0.0128851905, 0.0, 0.0, 0.0, -0.0143736629, -0.0389165916, -0.0102932947, -0.0032840597, 0.00192360999, -0.00428541843, -0.0277753733, 0.0267478805, -0.00404407736]), 17: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0353120416, 0.000296165817, 0.0, 0.0, 0.0, -0.0128851924, 0.0, 0.0, 0.0, 0.014373662, 0.0389165916, 0.0102932947, 0.00328406296, -0.00192360464, 0.00428541796, 0.0277753733, -0.0267478824, 0.00404408108]), 18: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0343304873, -0.0002846797, 0.0, 0.0, 0.0, 0.0126287891, 0.0, 0.0, 0.0, -0.0138563849, -0.038278047, -0.00892592221, -0.00309507037, 0.00181228935, -0.00364700635, -0.0272363164, 0.0356507674, 0.00630248059]), 19: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0343304873, 0.0002846797, 0.0, 0.0, 0.0, -0.0126287891, 0.0, 0.0, 0.0, 0.0138563868, 0.0382780507, 0.00892592501, 0.00309506967, -0.00181228423, 0.00364700728, 0.0272363182, -0.0356507674, -0.00630247965]), 20: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0334200077, -0.000273639365, 0.0, 0.0, 0.0, 0.0128525933, 0.0, 0.0, 0.0, 0.0, -0.0369335935, -0.00922013912, -0.00291697471, 0.00170741347, -0.00335700857, -0.0263028536, 0.0135796582, -0.0274060667, 0.0324016437, 0.0102990298]), 21: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.033420004, 0.0002736405, 0.0, 0.0, 0.0, -0.0128525943, 0.0, 0.0, 0.0, 0.0, 0.0369335935, 0.00922013819, 0.00291697239, -0.00170741463, 0.00335701043, 0.0263028499, -0.0135796601, 0.0274060667, -0.0324016437, -0.0102990288]), 22: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.032571841, -0.000263026246, 0.0, 0.0, 0.0, 0.0125917373, 0.0, 0.0, 0.0, -0.0137686618, -0.0363912843, -0.00795311201, -0.00274913176, 0.00160861027, -0.00277782138, -0.0258260444, 0.0337937325, 0.00584513275]), 23: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.032571841, 0.000263026246, 0.0, 0.0, 0.0, -0.0125917401, 0.0, 0.0, 0.0, 0.0137686618, 0.0363912806, 0.00795311388, 0.00274913432, -0.00160861376, 0.00277781999, 0.0258260444, -0.0337937325, -0.00584513275]), 24: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.031778533, -0.000252826489, 0.0, 0.0, 0.0, 0.0127925705, 0.0, 0.0, 0.0, 0.0, -0.0352173746, -0.00824390259, -0.00259095081, 0.001515529, -0.0212190133, 0.00195115269, 0.012833301, -0.0268460847, 0.0307130814, 0.00965970661]), 25: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.031778533, 0.000252824189, 0.0, 0.0, 0.0, -0.0127925714, 0.0, 0.0, 0.0, 0.0, 0.0352173746, 0.00824390166, 0.00259094918, -0.00151553017, 0.0212190133, -0.0019511549, -0.0128333019, 0.0268460829, -0.0307130814, -0.00965970568]), 26: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0310336929, -0.000243020419, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000564874441, -0.00998799503, -0.00588178029, 0.0053854771, 0.00242740219, 0.0256928708, -0.042142354, -0.00964517798, 0.00756192906, -0.0349458456, 0.00222304394, 0.0390917435, 0.0222816039, -0.035398826, -0.0427254587, -0.0128962072]), 27: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0310336929, 0.000243021554, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.000564875139, 0.0099879941, 0.00588177983, -0.00538547803, -0.00242740172, -0.0256928708, 0.042142354, 0.00964517705, -0.00756192999, 0.0349458419, -0.0022230444, -0.0390917435, -0.0222816039, 0.035398826, 0.0427254587, 0.0128962081]), 28: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0303318258, -0.000233597559, 0.0, 0.0, 0.0, 0.0121060284, 0.0, 0.0, 0.0, -0.0138423881, -0.034622401, -0.00698943529, -0.00234105485, 0.00138699648, -0.00197274354, -0.0246584024, 0.0321934335, 0.00539426459]), 29: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0303318296, 0.00023359523, 0.0, 0.0, 0.0, -0.0121060293, 0.0, 0.0, 0.0, 0.0138423881, 0.0346223973, 0.00698943296, 0.00234105322, -0.00138699473, 0.00197274378, 0.0246584006, -0.0321934335, -0.00539426459]), 30: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0296682045, -0.000224537056, 0.0, 0.0, 0.0, 0.0123991091, 0.0, 0.0, 0.0, 0.0, -0.0336812399, -0.00722563965, -0.00099554332, -0.0256339964, 0.00109576993, 0.020133514, 0.0121115278, -0.0263560545, 0.0291621368, 0.00907041132]), 31: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0296682045, 0.000224538206, 0.0, 0.0, 0.0, -0.0123991081, 0.0, 0.0, 0.0, 0.0, 0.0336812399, 0.00722564105, 0.000995542854, 0.0256339964, -0.00109577063, -0.020133514, -0.0121115278, 0.0263560526, -0.0291621368, -0.00907041226]), 32: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0290387068, -0.000215828521, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00432815868, 0.00960175041, 0.0125259198, -0.0019741077, 0.00230931211, 0.0242536534, -0.0404578149, -0.00877589732, 0.00713968603, -0.0337931477, 0.0313314497, -0.0117176045, 0.0212049317, -0.0342057385, -0.0414380617, -0.0117874108]), 33: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0290387068, 0.000215830834, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00432815868, -0.00960175041, -0.0125259198, 0.00197410793, -0.00230931072, -0.0242536552, 0.0404578112, 0.00877589826, -0.00713968836, 0.0337931477, -0.031331446, 0.0117176026, -0.0212049317, 0.0342057422, 0.0414380617, 0.0117874108])}
right_children_dict = {0: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, 27, -1, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 1: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, 27, -1, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 2: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, -1, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 3: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, -1, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 4: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, -1, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 5: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, -1, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 6: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, -1, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 7: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, -1, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 8: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, -1, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 9: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, -1, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 10: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, -1, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 11: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, -1, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 12: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 13: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 14: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 15: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 16: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 17: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 18: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 19: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 20: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 21: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 22: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 23: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 24: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 25: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 26: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 27: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 28: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 29: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 30: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 31: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, -1, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 32: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 33: np.array([1, 3, 5, 7, 9, -1, -1, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])}
split_feats_dict = {0: np.array([0, 4, 6, 10, 7, 0, 0, 9, 16, 16, 12, 18, 5, 9, 11, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1: np.array([0, 4, 6, 10, 7, 0, 0, 9, 16, 16, 12, 18, 5, 9, 11, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2: np.array([0, 4, 15, 10, 7, 0, 0, 9, 16, 16, 12, 18, 5, 9, 11, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 3: np.array([0, 4, 15, 10, 7, 0, 0, 9, 16, 16, 12, 18, 5, 9, 11, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4: np.array([0, 4, 15, 10, 7, 0, 0, 9, 16, 16, 12, 12, 5, 9, 11, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 5: np.array([0, 4, 15, 10, 7, 0, 0, 9, 16, 16, 12, 12, 5, 9, 11, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6: np.array([0, 4, 15, 10, 7, 0, 0, 9, 16, 16, 12, 18, 5, 9, 11, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7: np.array([0, 4, 15, 10, 7, 0, 0, 9, 16, 16, 12, 18, 5, 9, 11, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 8: np.array([0, 4, 15, 10, 7, 0, 0, 9, 16, 16, 12, 12, 5, 15, 11, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9: np.array([0, 4, 15, 10, 7, 0, 0, 9, 16, 16, 12, 12, 5, 15, 11, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10: np.array([0, 4, 15, 10, 7, 0, 0, 9, 16, 16, 12, 18, 5, 15, 11, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 11: np.array([0, 4, 15, 10, 7, 0, 0, 9, 16, 16, 12, 18, 5, 15, 11, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 12: np.array([0, 1, 15, 4, 4, 0, 0, 10, 7, 8, 0, 9, 16, 16, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 13: np.array([0, 1, 15, 4, 4, 0, 0, 10, 7, 8, 0, 9, 16, 16, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 14: np.array([0, 1, 15, 4, 6, 0, 0, 10, 7, 19, 0, 9, 16, 16, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 15: np.array([0, 1, 15, 4, 6, 0, 0, 10, 7, 19, 0, 9, 16, 16, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 16: np.array([0, 1, 15, 4, 4, 0, 0, 10, 13, 19, 0, 9, 7, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 17: np.array([0, 1, 15, 4, 4, 0, 0, 10, 13, 19, 0, 9, 7, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 18: np.array([0, 1, 15, 4, 6, 0, 0, 10, 13, 12, 0, 9, 15, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 19: np.array([0, 1, 15, 4, 6, 0, 0, 10, 13, 12, 0, 9, 15, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 20: np.array([0, 1, 15, 4, 4, 0, 0, 10, 7, 19, 0, 9, 7, 16, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 21: np.array([0, 1, 15, 4, 4, 0, 0, 10, 7, 19, 0, 9, 7, 16, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 22: np.array([0, 1, 15, 4, 6, 0, 0, 10, 13, 12, 0, 9, 15, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 23: np.array([0, 1, 15, 4, 6, 0, 0, 10, 13, 12, 0, 9, 15, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 24: np.array([0, 1, 15, 4, 4, 0, 0, 10, 7, 19, 0, 9, 16, 16, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 25: np.array([0, 1, 15, 4, 4, 0, 0, 10, 7, 19, 0, 9, 16, 16, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 26: np.array([0, 7, 15, 9, 9, 0, 0, 6, 2, 8, 19, 7, 3, 16, 19, 12, 9, 18, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 27: np.array([0, 7, 15, 9, 9, 0, 0, 6, 2, 8, 19, 7, 3, 16, 19, 12, 9, 18, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 28: np.array([0, 1, 15, 4, 6, 0, 0, 10, 13, 12, 0, 9, 7, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 29: np.array([0, 1, 15, 4, 6, 0, 0, 10, 13, 12, 0, 9, 7, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 30: np.array([0, 1, 15, 4, 4, 0, 0, 12, 7, 8, 0, 19, 19, 16, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 31: np.array([0, 1, 15, 4, 4, 0, 0, 12, 7, 8, 0, 19, 19, 16, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 32: np.array([0, 7, 15, 9, 9, 0, 0, 11, 2, 8, 19, 12, 2, 16, 19, 12, 8, 18, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 33: np.array([0, 7, 15, 9, 9, 0, 0, 11, 2, 8, 19, 12, 2, 16, 19, 12, 8, 18, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
split_vals_dict = {0: np.array([0.5, 1.5, 1.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1: np.array([0.5, 1.5, 1.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2: np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3: np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 4: np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 5: np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 6: np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 7: np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 8: np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 9: np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 10: np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 11: np.array([0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 12: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 13: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 14: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 15: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 16: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 17: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 18: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 19: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 20: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 21: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 22: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 23: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 24: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 25: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 26: np.array([0.5, 1.5, 0.5, 1.5, 1.5, 0.0, 0.0, 0.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 27: np.array([0.5, 1.5, 0.5, 1.5, 1.5, 0.0, 0.0, 0.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 28: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 29: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 30: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 1.5, 1.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 31: np.array([0.5, 1.5, 0.5, 1.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 1.5, 1.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 32: np.array([0.5, 1.5, 0.5, 1.5, 1.5, 0.0, 0.0, 1.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 33: np.array([0.5, 1.5, 0.5, 1.5, 1.5, 0.0, 0.0, 1.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}


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

    logits = [__build_logit_func(34, clss) for clss in range(n_classes)]

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
