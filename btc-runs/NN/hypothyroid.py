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
# Invocation: brainome TRAIN_TEST_SPLITS/hypothyroid-clean-train.csv -f NN -y -split 70 -modelonly -q -o btc-runs/NN/hypothyroid.py -json btc-runs/NN/hypothyroid.json
# Total compiler execution time: 0:00:26.23. Finished on: Feb-26-2022 18:42:18.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/NN/hypothyroid.py
    Classifier Type:              Neural Network
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        92.27%
      Training accuracy:          98.10% (1812/1847 correct)
      Validation Accuracy:        97.98% (777/793 correct)
      Combined Model Accuracy:    98.06% (2589/2640 correct)


    Model Capacity (MEC):         63    bits
    Generalization Ratio:         11.25 bits/bit
    Percent of Data Memorized:    24.47%
    Resilience to Noise:          -1.46 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |  1688    17 
                   1 |    18   124 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |   716    15 
                   1 |     1    61 

    Training Accuracy by Class:
         binaryClass |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
         ----------- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |  1688    18   124    17   99.00%   87.32%   98.94%   87.94%   98.97%   97.97%
                   1 |   124    17  1688    18   87.32%   99.00%   87.94%   98.94%   87.63%   77.99%

    Validation Accuracy by Class:
         binaryClass |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
         ----------- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |   716     1    61    15   97.95%   98.39%   99.86%   80.26%   98.90%   97.81%
                   1 |    61    15   716     1   98.39%   97.95%   80.26%   99.86%   88.41%   79.22%




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

IOBUF = 100000000
sys.setrecursionlimit(1000000)
TRAINFILE = ['TRAIN_TEST_SPLITS/hypothyroid-clean-train.csv']
mapping = {'0': 0, '1': 1}
ignorelabels = []
ignorecolumns = []
list_of_cols_to_normalize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
column_mappings = [{1.0: 0, 5.0: 1, 7.0: 2, 10.0: 3, 11.0: 4, 12.0: 5, 13.0: 6, 14.0: 7, 15.0: 8, 16.0: 9, 17.0: 10, 18.0: 11, 19.0: 12, 20.0: 13, 21.0: 14, 22.0: 15, 23.0: 16, 24.0: 17, 25.0: 18, 26.0: 19, 27.0: 20, 28.0: 21, 29.0: 22, 30.0: 23, 31.0: 24, 32.0: 25, 33.0: 26, 34.0: 27, 35.0: 28, 36.0: 29, 37.0: 30, 38.0: 31, 39.0: 32, 40.0: 33, 41.0: 34, 42.0: 35, 43.0: 36, 44.0: 37, 45.0: 38, 46.0: 39, 47.0: 40, 48.0: 41, 49.0: 42, 50.0: 43, 51.0: 44, 52.0: 45, 53.0: 46, 54.0: 47, 55.0: 48, 56.0: 49, 57.0: 50, 58.0: 51, 59.0: 52, 60.0: 53, 61.0: 54, 62.0: 55, 63.0: 56, 64.0: 57, 65.0: 58, 66.0: 59, 67.0: 60, 68.0: 61, 69.0: 62, 70.0: 63, 71.0: 64, 72.0: 65, 73.0: 66, 74.0: 67, 75.0: 68, 76.0: 69, 77.0: 70, 78.0: 71, 79.0: 72, 80.0: 73, 81.0: 74, 82.0: 75, 83.0: 76, 84.0: 77, 85.0: 78, 86.0: 79, 87.0: 80, 88.0: 81, 89.0: 82, 90.0: 83, 92.0: 84, 94.0: 85, 1684325040.0: 86, 6.0: 87, 8.0: 88, 93.0: 89, 91.0: 90, 2.0: 91}, {1304234792.0: 0, 1684325040.0: 1, 3664761504.0: 2}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {0.005: 0, 0.01: 1, 0.015: 2, 0.02: 3, 0.025: 4, 0.03: 5, 0.035: 6, 0.04: 7, 0.045: 8, 0.05: 9, 0.055: 10, 0.06: 11, 0.065: 12, 0.07: 13, 0.08: 14, 0.09: 15, 0.1: 16, 0.12: 17, 0.13: 18, 0.14: 19, 0.15: 20, 0.16: 21, 0.19: 22, 0.2: 23, 0.21: 24, 0.22: 25, 0.23: 26, 0.24: 27, 0.25: 28, 0.26: 29, 0.27: 30, 0.28: 31, 0.29: 32, 0.3: 33, 0.31: 34, 0.32: 35, 0.33: 36, 0.34: 37, 0.35: 38, 0.36: 39, 0.37: 40, 0.38: 41, 0.39: 42, 0.4: 43, 0.41: 44, 0.42: 45, 0.43: 46, 0.44: 47, 0.45: 48, 0.46: 49, 0.47: 50, 0.48: 51, 0.49: 52, 0.5: 53, 0.51: 54, 0.53: 55, 0.54: 56, 0.55: 57, 0.56: 58, 0.57: 59, 0.58: 60, 0.59: 61, 0.6: 62, 0.61: 63, 0.62: 64, 0.63: 65, 0.64: 66, 0.65: 67, 0.66: 68, 0.67: 69, 0.68: 70, 0.69: 71, 0.7: 72, 0.71: 73, 0.72: 74, 0.73: 75, 0.74: 76, 0.75: 77, 0.76: 78, 0.77: 79, 0.78: 80, 0.79: 81, 0.8: 82, 0.81: 83, 0.82: 84, 0.83: 85, 0.84: 86, 0.85: 87, 0.86: 88, 0.87: 89, 0.88: 90, 0.89: 91, 0.9: 92, 0.91: 93, 0.92: 94, 0.93: 95, 0.94: 96, 0.95: 97, 0.96: 98, 0.97: 99, 0.98: 100, 0.99: 101, 1.0: 102, 1.01: 103, 1.1: 104, 1.2: 105, 1.3: 106, 1.4: 107, 1.5: 108, 1.6: 109, 1.7: 110, 1.8: 111, 1.9: 112, 2.0: 113, 2.1: 114, 2.2: 115, 2.3: 116, 2.4: 117, 2.5: 118, 2.6: 119, 2.7: 120, 2.8: 121, 2.9: 122, 3.0: 123, 3.1: 124, 3.2: 125, 3.3: 126, 3.4: 127, 3.5: 128, 3.6: 129, 3.7: 130, 3.8: 131, 3.9: 132, 4.0: 133, 4.1: 134, 4.2: 135, 4.3: 136, 4.4: 137, 4.5: 138, 4.6: 139, 4.7: 140, 4.8: 141, 4.9: 142, 5.0: 143, 5.1: 144, 5.2: 145, 5.3: 146, 5.4: 147, 5.5: 148, 5.6: 149, 5.7: 150, 5.73: 151, 5.8: 152, 5.9: 153, 6.1: 154, 6.2: 155, 6.3: 156, 6.5: 157, 6.6: 158, 6.7: 159, 6.8: 160, 6.9: 161, 7.0: 162, 7.1: 163, 7.2: 164, 7.3: 165, 7.4: 166, 7.5: 167, 7.6: 168, 7.8: 169, 7.9: 170, 8.0: 171, 8.1: 172, 8.2: 173, 8.3: 174, 8.4: 175, 8.5: 176, 8.8: 177, 8.9: 178, 9.0: 179, 9.2: 180, 9.3: 181, 9.4: 182, 9.7: 183, 9.8: 184, 9.9: 185, 10.0: 186, 10.3: 187, 11.0: 188, 12.0: 189, 12.1: 190, 13.0: 191, 14.0: 192, 15.0: 193, 16.0: 194, 17.0: 195, 18.0: 196, 18.4: 197, 19.0: 198, 20.0: 199, 21.0: 200, 22.0: 201, 23.0: 202, 24.0: 203, 25.0: 204, 26.0: 205, 27.0: 206, 28.0: 207, 32.0: 208, 34.0: 209, 36.0: 210, 40.0: 211, 41.0: 212, 42.0: 213, 43.0: 214, 44.0: 215, 47.0: 216, 55.0: 217, 58.0: 218, 60.0: 219, 61.0: 220, 70.0: 221, 76.0: 222, 78.0: 223, 80.0: 224, 86.0: 225, 98.0: 226, 99.0: 227, 103.0: 228, 108.0: 229, 116.0: 230, 126.0: 231, 143.0: 232, 145.0: 233, 151.0: 234, 160.0: 235, 165.0: 236, 178.0: 237, 199.0: 238, 230.0: 239, 236.0: 240, 478.0: 241, 1684325040.0: 242, 0.17: 243, 6.0: 244, 9.5: 245, 7.7: 246, 0.52: 247, 54.0: 248, 188.0: 249, 11.4: 250, 89.0: 251, 468.0: 252, 35.0: 253, 82.0: 254, 65.0: 255, 6.4: 256, 29.0: 257, 183.0: 258, 8.6: 259, 530.0: 260, 46.0: 261, 39.0: 262, 109.0: 263, 11.1: 264}, {1993550816.0: 0, 2238339752.0: 1}, {0.05: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8, 1.0: 9, 1.1: 10, 1.2: 11, 1.3: 12, 1.4: 13, 1.44: 14, 1.5: 15, 1.6: 16, 1.7: 17, 1.8: 18, 1.9: 19, 2.0: 20, 2.1: 21, 2.2: 22, 2.3: 23, 2.4: 24, 2.5: 25, 2.6: 26, 2.7: 27, 2.8: 28, 2.9: 29, 3.0: 30, 3.1: 31, 3.2: 32, 3.3: 33, 3.4: 34, 3.5: 35, 3.6: 36, 3.7: 37, 3.8: 38, 3.9: 39, 4.0: 40, 4.1: 41, 4.2: 42, 4.3: 43, 4.4: 44, 4.5: 45, 4.6: 46, 4.7: 47, 4.8: 48, 5.0: 49, 5.1: 50, 5.2: 51, 5.3: 52, 5.4: 53, 5.5: 54, 5.7: 55, 6.2: 56, 6.6: 57, 6.7: 58, 7.1: 59, 7.6: 60, 10.6: 61, 1684325040.0: 62, 7.0: 63, 7.3: 64, 0.1: 65}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 2.9: 1, 3.0: 2, 4.0: 3, 4.8: 4, 5.8: 5, 10.0: 6, 11.0: 7, 12.0: 8, 13.0: 9, 14.0: 10, 15.0: 11, 16.0: 12, 17.0: 13, 19.0: 14, 22.0: 15, 23.0: 16, 25.0: 17, 27.0: 18, 29.0: 19, 31.0: 20, 32.0: 21, 33.0: 22, 35.0: 23, 36.0: 24, 37.0: 25, 38.0: 26, 39.0: 27, 41.0: 28, 42.0: 29, 43.0: 30, 44.0: 31, 45.0: 32, 46.0: 33, 48.0: 34, 49.0: 35, 50.0: 36, 51.0: 37, 52.0: 38, 53.0: 39, 54.0: 40, 55.0: 41, 56.0: 42, 57.0: 43, 58.0: 44, 59.0: 45, 60.0: 46, 61.0: 47, 62.0: 48, 63.0: 49, 64.0: 50, 65.0: 51, 66.0: 52, 67.0: 53, 68.0: 54, 69.0: 55, 70.0: 56, 71.0: 57, 72.0: 58, 73.0: 59, 74.0: 60, 75.0: 61, 76.0: 62, 77.0: 63, 78.0: 64, 79.0: 65, 80.0: 66, 81.0: 67, 82.0: 68, 83.0: 69, 84.0: 70, 85.0: 71, 86.0: 72, 87.0: 73, 88.0: 74, 89.0: 75, 90.0: 76, 91.0: 77, 92.0: 78, 93.0: 79, 94.0: 80, 95.0: 81, 96.0: 82, 97.0: 83, 98.0: 84, 99.0: 85, 100.0: 86, 101.0: 87, 102.0: 88, 103.0: 89, 104.0: 90, 105.0: 91, 106.0: 92, 107.0: 93, 108.0: 94, 109.0: 95, 110.0: 96, 111.0: 97, 112.0: 98, 113.0: 99, 114.0: 100, 115.0: 101, 116.0: 102, 117.0: 103, 118.0: 104, 119.0: 105, 120.0: 106, 121.0: 107, 122.0: 108, 123.0: 109, 124.0: 110, 125.0: 111, 126.0: 112, 127.0: 113, 128.0: 114, 129.0: 115, 130.0: 116, 131.0: 117, 132.0: 118, 133.0: 119, 134.0: 120, 135.0: 121, 136.0: 122, 137.0: 123, 138.0: 124, 139.0: 125, 140.0: 126, 141.0: 127, 142.0: 128, 143.0: 129, 144.0: 130, 145.0: 131, 146.0: 132, 147.0: 133, 148.0: 134, 149.0: 135, 151.0: 136, 152.0: 137, 153.0: 138, 154.0: 139, 155.0: 140, 156.0: 141, 157.0: 142, 158.0: 143, 159.0: 144, 160.0: 145, 161.0: 146, 162.0: 147, 163.0: 148, 164.0: 149, 165.0: 150, 166.0: 151, 167.0: 152, 168.0: 153, 170.0: 154, 171.0: 155, 172.0: 156, 173.0: 157, 174.0: 158, 175.0: 159, 176.0: 160, 177.0: 161, 178.0: 162, 179.0: 163, 180.0: 164, 181.0: 165, 182.0: 166, 183.0: 167, 184.0: 168, 187.0: 169, 189.0: 170, 191.0: 171, 192.0: 172, 193.0: 173, 194.0: 174, 196.0: 175, 197.0: 176, 198.0: 177, 200.0: 178, 203.0: 179, 204.0: 180, 206.0: 181, 210.0: 182, 211.0: 183, 212.0: 184, 213.0: 185, 217.0: 186, 219.0: 187, 222.0: 188, 223.0: 189, 226.0: 190, 230.0: 191, 231.0: 192, 233.0: 193, 235.0: 194, 239.0: 195, 240.0: 196, 244.0: 197, 248.0: 198, 250.0: 199, 255.0: 200, 258.0: 201, 261.0: 202, 263.0: 203, 272.0: 204, 289.0: 205, 301.0: 206, 372.0: 207, 430.0: 208, 1684325040.0: 209, 205.0: 210, 252.0: 211, 188.0: 212, 201.0: 213, 150.0: 214, 256.0: 215, 237.0: 216, 169.0: 217, 257.0: 218, 21.0: 219, 28.0: 220, 47.0: 221}, {1993550816.0: 0, 2238339752.0: 1}, {0.36: 0, 0.38: 1, 0.41: 2, 0.46: 3, 0.48: 4, 0.49: 5, 0.5: 6, 0.52: 7, 0.53: 8, 0.54: 9, 0.56: 10, 0.58: 11, 0.59: 12, 0.6: 13, 0.61: 14, 0.62: 15, 0.64: 16, 0.65: 17, 0.66: 18, 0.67: 19, 0.68: 20, 0.69: 21, 0.7: 22, 0.71: 23, 0.72: 24, 0.73: 25, 0.74: 26, 0.75: 27, 0.76: 28, 0.77: 29, 0.78: 30, 0.79: 31, 0.8: 32, 0.81: 33, 0.82: 34, 0.83: 35, 0.84: 36, 0.85: 37, 0.86: 38, 0.87: 39, 0.88: 40, 0.89: 41, 0.9: 42, 0.91: 43, 0.92: 44, 0.93: 45, 0.94: 46, 0.944: 47, 0.95: 48, 0.96: 49, 0.97: 50, 0.98: 51, 0.99: 52, 1.0: 53, 1.01: 54, 1.02: 55, 1.03: 56, 1.04: 57, 1.05: 58, 1.06: 59, 1.07: 60, 1.08: 61, 1.09: 62, 1.1: 63, 1.11: 64, 1.12: 65, 1.13: 66, 1.14: 67, 1.15: 68, 1.16: 69, 1.17: 70, 1.18: 71, 1.19: 72, 1.2: 73, 1.21: 74, 1.22: 75, 1.23: 76, 1.24: 77, 1.25: 78, 1.26: 79, 1.27: 80, 1.28: 81, 1.29: 82, 1.3: 83, 1.31: 84, 1.32: 85, 1.33: 86, 1.34: 87, 1.35: 88, 1.36: 89, 1.38: 90, 1.39: 91, 1.4: 92, 1.41: 93, 1.42: 94, 1.43: 95, 1.44: 96, 1.46: 97, 1.47: 98, 1.48: 99, 1.49: 100, 1.5: 101, 1.51: 102, 1.52: 103, 1.53: 104, 1.54: 105, 1.55: 106, 1.56: 107, 1.57: 108, 1.58: 109, 1.59: 110, 1.62: 111, 1.63: 112, 1.65: 113, 1.66: 114, 1.67: 115, 1.68: 116, 1.69: 117, 1.7: 118, 1.71: 119, 1.73: 120, 1.74: 121, 1.75: 122, 1.76: 123, 1.77: 124, 1.8: 125, 1.82: 126, 1.83: 127, 1.88: 128, 1.93: 129, 1.94: 130, 2.01: 131, 2.03: 132, 2.12: 133, 2.32: 134, 1684325040.0: 135, 1.45: 136, 1.61: 137, 1.97: 138, 0.63: 139, 1.79: 140, 1.37: 141}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 2.8: 1, 3.0: 2, 4.0: 3, 5.4: 4, 7.0: 5, 8.4: 6, 8.9: 7, 10.0: 8, 11.0: 9, 13.0: 10, 14.0: 11, 17.0: 12, 19.0: 13, 20.0: 14, 24.0: 15, 26.0: 16, 27.0: 17, 32.0: 18, 33.0: 19, 34.0: 20, 36.0: 21, 37.0: 22, 39.0: 23, 42.0: 24, 47.0: 25, 49.0: 26, 50.0: 27, 51.0: 28, 52.0: 29, 53.0: 30, 54.0: 31, 55.0: 32, 56.0: 33, 58.0: 34, 59.0: 35, 60.0: 36, 61.0: 37, 62.0: 38, 63.0: 39, 64.0: 40, 65.0: 41, 66.0: 42, 67.0: 43, 68.0: 44, 69.0: 45, 70.0: 46, 71.0: 47, 72.0: 48, 73.0: 49, 74.0: 50, 75.0: 51, 76.0: 52, 77.0: 53, 78.0: 54, 79.0: 55, 80.0: 56, 81.0: 57, 82.0: 58, 83.0: 59, 84.0: 60, 85.0: 61, 86.0: 62, 87.0: 63, 88.0: 64, 89.0: 65, 90.0: 66, 91.0: 67, 92.0: 68, 93.0: 69, 94.0: 70, 95.0: 71, 96.0: 72, 97.0: 73, 98.0: 74, 99.0: 75, 100.0: 76, 101.0: 77, 102.0: 78, 103.0: 79, 104.0: 80, 105.0: 81, 106.0: 82, 107.0: 83, 108.0: 84, 109.0: 85, 110.0: 86, 111.0: 87, 112.0: 88, 113.0: 89, 114.0: 90, 115.0: 91, 116.0: 92, 117.0: 93, 118.0: 94, 119.0: 95, 120.0: 96, 121.0: 97, 122.0: 98, 123.0: 99, 124.0: 100, 125.0: 101, 126.0: 102, 127.0: 103, 128.0: 104, 129.0: 105, 130.0: 106, 131.0: 107, 132.0: 108, 133.0: 109, 134.0: 110, 135.0: 111, 136.0: 112, 137.0: 113, 138.0: 114, 139.0: 115, 140.0: 116, 141.0: 117, 142.0: 118, 143.0: 119, 144.0: 120, 145.0: 121, 146.0: 122, 147.0: 123, 148.0: 124, 149.0: 125, 150.0: 126, 151.0: 127, 152.0: 128, 153.0: 129, 154.0: 130, 155.0: 131, 156.0: 132, 157.0: 133, 158.0: 134, 159.0: 135, 160.0: 136, 161.0: 137, 162.0: 138, 163.0: 139, 164.0: 140, 165.0: 141, 166.0: 142, 167.0: 143, 168.0: 144, 169.0: 145, 170.0: 146, 171.0: 147, 172.0: 148, 173.0: 149, 175.0: 150, 176.0: 151, 177.0: 152, 178.0: 153, 179.0: 154, 180.0: 155, 182.0: 156, 183.0: 157, 185.0: 158, 186.0: 159, 188.0: 160, 189.0: 161, 190.0: 162, 191.0: 163, 195.0: 164, 196.0: 165, 198.0: 166, 199.0: 167, 200.0: 168, 201.0: 169, 203.0: 170, 204.0: 171, 207.0: 172, 209.0: 173, 214.0: 174, 215.0: 175, 216.0: 176, 217.0: 177, 218.0: 178, 219.0: 179, 220.0: 180, 221.0: 181, 222.0: 182, 223.0: 183, 224.0: 184, 227.0: 185, 242.0: 186, 244.0: 187, 247.0: 188, 251.0: 189, 253.0: 190, 274.0: 191, 280.0: 192, 281.0: 193, 291.0: 194, 349.0: 195, 362.0: 196, 395.0: 197, 1684325040.0: 198, 174.0: 199, 194.0: 200, 235.0: 201, 35.0: 202, 197.0: 203, 283.0: 204, 184.0: 205, 46.0: 206, 9.0: 207, 28.0: 208, 48.0: 209, 8.5: 210, 43.0: 211, 41.0: 212}, {1993550816.0: 0}, {1684325040.0: 0}, {596708387.0: 0, 1203304565.0: 1, 1918519837.0: 2, 3646436640.0: 3, 3655101910.0: 4}]
target = ''
target_column = 29
important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
ignore_idxs = []
classifier_type = 'NN'
num_attr = 29
n_classes = 2
model_cap = 63
w_h = np.array([[-0.29453161358833313, 0.035298723727464676, -0.49272996187210083, -0.24225877225399017, -0.10372237861156464, 0.06850244849920273, 0.07417832314968109, 0.28822481632232666, -0.098241426050663, -0.08363627642393112, -0.2906635105609894, -0.17758604884147644, -0.3576664924621582, -0.3366524577140808, -0.12010668963193893, 0.023053620010614395, -0.11767566949129105, -0.03580882027745247, -0.5861156582832336, -0.455044686794281, -0.13181401789188385, 0.08850256353616714, -0.38792693614959717, 0.05404912307858467, -0.36829811334609985, -0.21669462323188782, 0.39835643768310547, -0.4081205427646637, -0.5659589767456055], [0.020716823637485504, 0.4284500777721405, 10.181435585021973, 2.318638801574707, 11.517692565917969, 1.6158277988433838, 9.894432067871094, 8.329516410827637, -0.014930473640561104, 0.12976808845996857, -0.2550097405910492, -0.00573775265365839, 12.755542755126953, -0.39514586329460144, 1.001825213432312, -0.21381178498268127, -16.901601791381836, -0.1270027756690979, 8.153249740600586, 0.1813385933637619, 1.476449728012085, 0.007293002679944038, 5.470207214355469, 0.013620581477880478, 5.092280387878418, 0.0641557052731514, -0.259038507938385, 0.08198044449090958, 0.059074632823467255]])
b_h = np.array([-0.33090776205062866, 9.24284839630127])
w_o = np.array([[-0.7329521179199219, -0.8140685558319092]])
b_o = np.array(2.707915782928467)


class PredictorError(Exception):

    def __init__(self, msg, code):
        self.msg = msg
        self.code = code

    def __str__(self):
        return self.msg
def __column_norm(column, mappings):
    normalized_col = np.zeros(column.shape[0])
    for i, val in enumerate(column.reshape(-1)):
        if val not in mappings:
            mappings[val] = int(max(mappings.values())) + 1
        normalized_col[i] = mappings[val]
    return normalized_col


def __normalize(arr):
    for i,mapping in zip(list_of_cols_to_normalize, column_mappings):
        if i >= arr.shape[1]:
            break
        col = arr[:, i]
        normcol = __column_norm(col, mapping)
        arr[:, i] = normcol
    return arr


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


def __classify(arr, return_probabilities=False):
    h = np.dot(arr, w_h.T) + b_h
    relu = np.maximum(h, np.zeros_like(h))
    out = np.dot(relu, w_o.T) + b_o
    if return_probabilities:
        exp_o = np.zeros((out.shape[0],))
        idxs_negative = np.argwhere(out < 0.).reshape(-1)
        if idxs_negative.shape[0] > 0:
            exp_o[idxs_negative] = 1. - 1. / (1. + np.exp(out[idxs_negative])).reshape(-1)
        idxs_positive = np.argwhere(out >= 0.).reshape(-1)
        if idxs_positive.shape[0] > 0:
            exp_o[idxs_positive] = 1. / (1. + np.exp(-out[idxs_positive])).reshape(-1)
        exp_o = exp_o.reshape(-1, 1)
        output = np.concatenate((1. - exp_o, exp_o), axis=1)
    else:
        output = (out >= 0).astype('int').reshape(-1)
    return output



def __validate_kwargs(kwargs):
    for key in kwargs:

        if key not in ['return_probabilities']:
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

    arr = __normalize(arr)

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
            print("Classifier Type:                    Neural Network")

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
