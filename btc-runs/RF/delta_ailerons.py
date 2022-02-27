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
# Invocation: brainome TRAIN_TEST_SPLITS/delta_ailerons-clean-train.csv -f RF -y -split 70 -modelonly -q -o btc-runs/RF/delta_ailerons.py -json btc-runs/RF/delta_ailerons.json
# Total compiler execution time: 0:00:06.20. Finished on: Feb-26-2022 18:28:56.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/RF/delta_ailerons.py
    Classifier Type:              Random Forest
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        53.07%
      Training accuracy:          96.13% (3357/3492 correct)
      Validation Accuracy:        94.65% (1418/1498 correct)
      Combined Model Accuracy:    95.69% (4775/4990 correct)


    Model Capacity (MEC):         56    bits
    Generalization Ratio:         59.78 bits/bit
    Percent of Data Memorized:     3.35%
    Resilience to Noise:          -1.78 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |  1793    60 
                   1 |    75  1564 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |   759    36 
                   1 |    44   659 

    Training Accuracy by Class:
         binaryClass |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
         ----------- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |  1793    75  1564    60   96.76%   95.42%   95.99%   96.31%   96.37%   93.00%
                   1 |  1564    60  1793    75   95.42%   96.76%   96.31%   95.99%   95.86%   92.05%

    Validation Accuracy by Class:
         binaryClass |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
         ----------- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |   759    44   659    36   95.47%   93.74%   94.52%   94.82%   94.99%   90.46%
                   1 |   659    36   759    44   93.74%   95.47%   94.82%   94.52%   94.28%   89.17%


    Attribute Ranking:
                                      Feature | Relative Importance
                                     RollRate :   0.7735
                                     currRoll :   0.0725
                                 diffRollRate :   0.0671
                                    currPitch :   0.0492
                                    PitchRate :   0.0377
         

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
TRAINFILE = ['TRAIN_TEST_SPLITS/delta_ailerons-clean-train.csv']
mapping = {'0': 0, '1': 1}
ignorelabels = []
ignorecolumns = []
target = ''
target_column = 5
important_idxs = [0, 1, 2, 3, 4]
ignore_idxs = []
classifier_type = 'RF'
num_attr = 5
n_classes = 2
model_cap = 56
logits_dict = {0: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.477292329, -0.851377547, -0.791850626, -0.27700001, -0.709119976, -0.204553843, 0.709119976, -0.413653314, -0.126628578, 0.590933323, -0.531840026, 0.354559988, -0.633142889, 0.738666654, 0.793256342, -0.379885703]), 1: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.477292329, 0.851377547, 0.791850626, 0.27700001, 0.709119976, 0.204553843, -0.709119976, 0.413653314, 0.126628578, -0.590933323, 0.531840026, -0.354559988, 0.633142889, -0.738666654, -0.793256342, 0.379885703]), 2: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0882990062, -0.467632562, 0.404229671, -0.46124354, -0.170221165, 0.867662907, -0.432584316, -0.179164499, -1.05491173, 0.00344941765, -0.00291656051, 0.395321906, -0.552829623, 0.266394377, 0.488312781, 0.360596538]), 3: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0882990286, 0.467632562, -0.404229611, 0.4612436, 0.170221105, -0.867662907, 0.432584286, 0.179164499, 1.05491161, -0.00344955083, 0.00291660195, -0.395321965, 0.552829742, -0.266394347, -0.488312781, -0.360596597]), 4: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.372311115, 0.353134006, -0.444812208, -0.292109221, -0.286707371, 0.14201884, 0.831519365, -0.0712158456, -0.066921033, 0.527347445, 0.130719289, 0.382504612, -0.331874162, 0.541616082, -0.48823753, 0.244974598]), 5: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.372311115, -0.353133887, 0.444812208, 0.292109311, 0.286707342, -0.142018944, -0.831519127, 0.0712159798, 0.0669210777, -0.527347445, -0.130719364, -0.382504672, 0.331874043, -0.541616082, 0.48823747, -0.244974598]), 6: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.312447131, -0.0969489515, -1.18173897, -0.0975017846, -0.208874449, 0.508439004, 0.455807656, 0.00982980896, -0.476569444, 0.441397637, 0.400318235, 0.183181211, 0.448257476, -0.0555120185, 0.449647814, 0.255965382]), 7: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.312447071, 0.0969486609, 1.18173897, 0.0975017548, 0.208874345, -0.508439004, -0.455807567, -0.00982974563, 0.476569414, -0.441397667, -0.400318235, -0.183181196, -0.448257476, 0.0555119924, -0.449647814, -0.255965412]), 8: np.array([0.0, 0.0, 0.0, 0.755527318, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0629673526, -0.43422538, -0.209896237, 0.50919795, 0.127511814, -0.408583343, -0.0325650759, 0.638239741, 0.100704998, 0.39141953, 0.0782424808, -0.472069949]), 9: np.array([0.0, 0.0, 0.0, -0.755527318, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0629671142, 0.434225351, 0.209896222, -0.50919807, -0.127511978, 0.408583194, 0.03256502, -0.638239741, -0.10070499, -0.3914195, -0.0782424659, 0.472069979]), 10: np.array([0.0, -0.437347651, 0.0, 0.0, 0.0, 0.0939978063, 1.05714309, 0.0, 0.0, -0.399594516, 0.0135595137, 0.285515755, -0.417285174]), 11: np.array([0.0, 0.437347621, 0.0, 0.0, 0.0, -0.0939982682, -1.05714273, 0.0, 0.0, 0.399594516, -0.013559619, -0.285515666, 0.417285293]), 12: np.array([0.0, 0.0, 0.0, 0.0, -0.928273141, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00665616803, 0.0, -0.034329664, 0.243915617, -0.534970701, 0.273004532, 0.579231977, -0.0670083389, -0.678418458, 0.311000198, 0.135768622, 0.447465032]), 13: np.array([0.0, 0.0, 0.0, 0.0, 0.92827332, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00665627606, 0.0, 0.0343296416, -0.243915603, 0.534970701, -0.273004472, -0.579232216, 0.0670082718, 0.678418577, -0.311000258, -0.135768592, -0.447465032]), 14: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0710379556, -0.900733829, 0.313844979, 0.0328032859, 1.01343107, 0.419075727, -0.325002611, 0.213558078, -0.188462973, -0.71346271, -0.135629401, 0.391917855, -0.0270818695, -0.405482709, 0.0515995398, 0.415998787]), 15: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0710378587, 0.90073365, -0.313845038, -0.0328033566, -1.01343107, -0.419075727, 0.325002611, -0.213558123, 0.188463047, 0.713462651, 0.135629267, -0.391917944, 0.0270818584, 0.405482709, -0.0515995212, -0.415998787])}
right_children_dict = {0: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 1: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 2: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 3: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 4: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 5: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 6: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 7: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 8: np.array([1, 3, 5, -1, 7, 9, 11, 13, 15, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 9: np.array([1, 3, 5, -1, 7, 9, 11, 13, 15, 17, 19, 21, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 10: np.array([1, -1, 3, 5, 7, -1, -1, 9, 11, -1, -1, -1, -1]), 11: np.array([1, -1, 3, 5, 7, -1, -1, 9, 11, -1, -1, -1, -1]), 12: np.array([1, 3, 5, 7, -1, 9, 11, 13, 15, 17, 19, -1, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 13: np.array([1, 3, 5, 7, -1, 9, 11, 13, 15, 17, 19, -1, 21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 14: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), 15: np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])}
split_feats_dict = {0: np.array([0, 0, 4, 3, 3, 1, 0, 2, 4, 4, 3, 4, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1: np.array([0, 0, 4, 3, 3, 1, 0, 2, 4, 4, 3, 4, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 2: np.array([0, 3, 0, 0, 2, 3, 4, 1, 0, 0, 4, 4, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 3: np.array([0, 3, 0, 0, 2, 3, 4, 1, 0, 0, 4, 4, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 4: np.array([0, 0, 2, 1, 3, 4, 0, 3, 3, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 5: np.array([0, 0, 2, 1, 3, 4, 0, 3, 3, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 6: np.array([0, 3, 2, 0, 4, 4, 0, 0, 4, 0, 2, 3, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 7: np.array([0, 3, 2, 0, 4, 4, 0, 0, 4, 0, 2, 3, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 8: np.array([0, 2, 3, 0, 3, 2, 2, 3, 1, 3, 3, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 9: np.array([0, 2, 3, 0, 3, 2, 2, 3, 1, 3, 3, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 10: np.array([0, 0, 0, 3, 0, 0, 0, 3, 1, 0, 0, 0, 0]), 11: np.array([0, 0, 0, 3, 0, 0, 0, 3, 1, 0, 0, 0, 0]), 12: np.array([4, 4, 0, 3, 0, 2, 2, 3, 3, 4, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 13: np.array([4, 4, 0, 3, 0, 2, 2, 3, 3, 4, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 14: np.array([3, 3, 3, 4, 2, 4, 0, 0, 4, 0, 0, 4, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 15: np.array([3, 3, 3, 4, 2, 4, 0, 0, 4, 0, 0, 4, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
split_vals_dict = {0: np.array([-0.000250000012, -0.00164999999, -0.000195000001, 0.00850000046, 0.00900000054, 0.0051500001, 0.000349999988, 0.000500000024, -0.000104999999, 0.000119999997, 0.0299999993, -0.000490000006, 0.00554999989, 0.00249999971, 0.0304999985, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1: np.array([-0.000250000012, -0.00164999999, -0.000195000001, 0.00850000046, 0.00900000054, 0.0051500001, 0.000349999988, 0.000500000024, -0.000104999999, 0.000119999997, 0.0299999993, -0.000490000006, 0.00554999989, 0.00249999971, 0.0304999985, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2: np.array([-0.000349999988, 0.00650000013, 0.00524999993, -0.000950000016, 0.00449999981, -0.00850000046, -0.000375000003, -0.00439999998, -0.000750000007, -0.00655000005, -0.000165000005, 9.49999958e-05, -0.000155000002, 0.00590000022, 0.0135000004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 3: np.array([-0.000349999988, 0.00650000013, 0.00524999993, -0.000950000016, 0.00449999981, -0.00850000046, -0.000375000003, -0.00439999998, -0.000750000007, -0.00655000005, -0.000165000005, 9.49999958e-05, -0.000155000002, 0.00590000022, 0.0135000004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 4: np.array([-0.00195000006, -0.00444999989, 0.0135000004, -0.00215000007, 0.00850000046, -0.000195000001, 0.0051500001, 0.00249999994, 0.00650000013, 0.0185000002, 0.0125000002, 0.0434999987, -0.0075000003, 0.0265000015, -0.0265000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 5: np.array([-0.00195000006, -0.00444999989, 0.0135000004, -0.00215000007, 0.00850000046, -0.000195000001, 0.0051500001, 0.00249999994, 0.00650000013, 0.0185000002, 0.0125000002, 0.0434999987, -0.0075000003, 0.0265000015, -0.0265000015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 6: np.array([0.00295000011, 0.00949999969, 0.00949999969, 0.00164999999, -0.000165000005, -0.000135000009, 0.00744999992, -0.00435000006, 0.000119999997, 0.0023500002, 0.0104999999, 0.0265000015, 0.00175000005, 0.00345000019, 0.0164999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 7: np.array([0.00295000011, 0.00949999969, 0.00949999969, 0.00164999999, -0.000165000005, -0.000135000009, 0.00744999992, -0.00435000006, 0.000119999997, 0.0023500002, 0.0104999999, 0.0265000015, 0.00175000005, 0.00345000019, 0.0164999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 8: np.array([-0.00284999982, 0.000500000024, -0.0075000003, 0.0, 0.0254999995, 0.0175000001, 0.0185000002, 0.0164999999, 0.000750000007, -0.0125000002, -0.0125000002, 4.99999987e-06, 0.00270000007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 9: np.array([-0.00284999982, 0.000500000024, -0.0075000003, 0.0, 0.0254999995, 0.0175000001, 0.0185000002, 0.0164999999, 0.000750000007, -0.0125000002, -0.0125000002, 4.99999987e-06, 0.00270000007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 10: np.array([-0.00844999962, 0.0, -0.00815000013, 0.00650000013, 0.00674999971, 0.0, 0.0, -0.0254999995, 0.00564999972, 0.0, 0.0, 0.0, 0.0]), 11: np.array([-0.00844999962, 0.0, -0.00815000013, 0.00650000013, 0.00674999971, 0.0, 0.0, -0.0254999995, 0.00564999972, 0.0, 0.0, 0.0, 0.0]), 12: np.array([0.000235, 0.000224999996, 0.00384999998, 0.0354999974, 0.0, 0.0104999999, 0.00249999994, 0.0265000015, 0.0439999998, 0.000304999994, 0.0250000004, 0.0, 0.000264999981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 13: np.array([0.000235, 0.000224999996, 0.00384999998, 0.0354999974, 0.0, 0.0104999999, 0.00249999994, 0.0265000015, 0.0439999998, 0.000304999994, 0.0250000004, 0.0, 0.000264999981, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 14: np.array([0.0164999999, 0.0104999999, 0.0265000015, 4.99999987e-06, 0.0075000003, -0.000104999999, -0.00350000011, 0.00405000011, 4.50000007e-05, -0.00399999972, -0.00645000022, -0.000245000003, 0.0154999997, 0.00179999997, -0.000174999994, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 15: np.array([0.0164999999, 0.0104999999, 0.0265000015, 4.99999987e-06, 0.0075000003, -0.000104999999, -0.00350000011, 0.00405000011, 4.50000007e-05, -0.00399999972, -0.00645000022, -0.000245000003, 0.0154999997, 0.00179999997, -0.000174999994, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}


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

    logits = [__build_logit_func(16, clss) for clss in range(n_classes)]

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
