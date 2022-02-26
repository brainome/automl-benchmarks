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
# Invocation: brainome TRAIN_TEST_SPLITS/dataset_38_sick-clean-train.csv -f NN -y -split 70 -modelonly -q -o btc-runs/NN/sick.py -json btc-runs/NN/sick.json
# Total compiler execution time: 0:00:21.27. Finished on: Feb-26-2022 18:56:31.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/NN/sick.py
    Classifier Type:              Neural Network
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        93.86%
      Training accuracy:          96.64% (1785/1847 correct)
      Validation Accuracy:        96.59% (766/793 correct)
      Combined Model Accuracy:    96.62% (2551/2640 correct)


    Model Capacity (MEC):         53    bits
    Generalization Ratio:         11.18 bits/bit
    Percent of Data Memorized:    25.84%
    Resilience to Noise:          -1.53 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |  1716    18 
                   1 |    44    69 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |   738     6 
                   1 |    21    28 

    Training Accuracy by Class:
               Class |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |  1716    44    69    18   98.96%   61.06%   97.50%   79.31%   98.23%   96.51%
                   1 |    69    18  1716    44   61.06%   98.96%   79.31%   97.50%   69.00%   52.67%

    Validation Accuracy by Class:
               Class |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |   738    21    28     6   99.19%   57.14%   97.23%   82.35%   98.20%   96.47%
                   1 |    28     6   738    21   57.14%   99.19%   82.35%   97.23%   67.47%   50.91%




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
TRAINFILE = ['TRAIN_TEST_SPLITS/dataset_38_sick-clean-train.csv']
mapping = {'0': 0, '1': 1}
ignorelabels = []
ignorecolumns = []
list_of_cols_to_normalize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
column_mappings = [{1304234792.0: 0, 1684325040.0: 1, 3664761504.0: 2}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {1993550816.0: 0, 2238339752.0: 1}, {0.005: 0, 0.01: 1, 0.015: 2, 0.02: 3, 0.025: 4, 0.03: 5, 0.035: 6, 0.04: 7, 0.045: 8, 0.05: 9, 0.055: 10, 0.06: 11, 0.065: 12, 0.07: 13, 0.08: 14, 0.09: 15, 0.1: 16, 0.12: 17, 0.13: 18, 0.14: 19, 0.15: 20, 0.16: 21, 0.17: 22, 0.18: 23, 0.19: 24, 0.2: 25, 0.21: 26, 0.24: 27, 0.25: 28, 0.26: 29, 0.28: 30, 0.29: 31, 0.3: 32, 0.31: 33, 0.32: 34, 0.33: 35, 0.34: 36, 0.35: 37, 0.36: 38, 0.37: 39, 0.38: 40, 0.39: 41, 0.4: 42, 0.41: 43, 0.42: 44, 0.43: 45, 0.44: 46, 0.45: 47, 0.46: 48, 0.47: 49, 0.48: 50, 0.49: 51, 0.5: 52, 0.51: 53, 0.52: 54, 0.54: 55, 0.55: 56, 0.56: 57, 0.57: 58, 0.58: 59, 0.59: 60, 0.6: 61, 0.61: 62, 0.62: 63, 0.63: 64, 0.64: 65, 0.65: 66, 0.66: 67, 0.67: 68, 0.68: 69, 0.69: 70, 0.7: 71, 0.72: 72, 0.73: 73, 0.74: 74, 0.75: 75, 0.77: 76, 0.78: 77, 0.79: 78, 0.8: 79, 0.81: 80, 0.82: 81, 0.83: 82, 0.84: 83, 0.85: 84, 0.86: 85, 0.87: 86, 0.88: 87, 0.89: 88, 0.9: 89, 0.91: 90, 0.92: 91, 0.93: 92, 0.94: 93, 0.95: 94, 0.97: 95, 0.98: 96, 0.99: 97, 1.0: 98, 1.1: 99, 1.2: 100, 1.3: 101, 1.4: 102, 1.5: 103, 1.6: 104, 1.7: 105, 1.8: 106, 1.9: 107, 2.0: 108, 2.1: 109, 2.2: 110, 2.3: 111, 2.4: 112, 2.5: 113, 2.6: 114, 2.7: 115, 2.8: 116, 2.9: 117, 3.0: 118, 3.1: 119, 3.2: 120, 3.3: 121, 3.4: 122, 3.5: 123, 3.6: 124, 3.7: 125, 3.8: 126, 3.9: 127, 4.0: 128, 4.1: 129, 4.2: 130, 4.3: 131, 4.4: 132, 4.5: 133, 4.6: 134, 4.7: 135, 4.8: 136, 4.9: 137, 5.0: 138, 5.1: 139, 5.2: 140, 5.4: 141, 5.5: 142, 5.6: 143, 5.7: 144, 5.8: 145, 5.9: 146, 6.0: 147, 6.1: 148, 6.2: 149, 6.3: 150, 6.4: 151, 6.5: 152, 6.7: 153, 6.8: 154, 6.9: 155, 7.0: 156, 7.1: 157, 7.2: 158, 7.4: 159, 7.5: 160, 7.6: 161, 7.7: 162, 7.8: 163, 7.9: 164, 8.0: 165, 8.1: 166, 8.2: 167, 8.3: 168, 8.4: 169, 8.5: 170, 8.6: 171, 8.8: 172, 9.0: 173, 9.1: 174, 9.2: 175, 9.3: 176, 9.6: 177, 9.7: 178, 9.9: 179, 10.0: 180, 11.0: 181, 11.4: 182, 12.0: 183, 13.0: 184, 14.0: 185, 14.8: 186, 15.0: 187, 16.0: 188, 17.0: 189, 18.0: 190, 18.4: 191, 20.0: 192, 21.0: 193, 22.0: 194, 23.0: 195, 24.0: 196, 25.0: 197, 26.0: 198, 26.4: 199, 27.0: 200, 28.0: 201, 30.5: 202, 31.0: 203, 34.0: 204, 35.0: 205, 36.0: 206, 38.0: 207, 40.0: 208, 42.0: 209, 43.0: 210, 44.0: 211, 45.0: 212, 46.0: 213, 47.0: 214, 50.0: 215, 54.0: 216, 55.0: 217, 58.0: 218, 60.0: 219, 61.0: 220, 70.0: 221, 78.0: 222, 80.0: 223, 86.0: 224, 89.0: 225, 98.0: 226, 99.0: 227, 100.0: 228, 103.0: 229, 108.0: 230, 117.0: 231, 126.0: 232, 143.0: 233, 145.0: 234, 151.0: 235, 160.0: 236, 165.0: 237, 183.0: 238, 230.0: 239, 236.0: 240, 530.0: 241, 1684325040.0: 242, 7.3: 243, 0.76: 244, 188.0: 245, 10.3: 246, 6.6: 247, 0.96: 248, 8.9: 249, 19.0: 250, 41.0: 251, 0.22: 252, 139.0: 253, 0.53: 254, 5.3: 255, 9.4: 256, 0.27: 257, 400.0: 258, 76.0: 259, 65.0: 260, 5.73: 261, 0.71: 262}, {1993550816.0: 0, 2238339752.0: 1}, {0.05: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8, 1.0: 9, 1.1: 10, 1.2: 11, 1.3: 12, 1.4: 13, 1.44: 14, 1.5: 15, 1.6: 16, 1.7: 17, 1.8: 18, 1.9: 19, 2.0: 20, 2.1: 21, 2.2: 22, 2.3: 23, 2.4: 24, 2.5: 25, 2.6: 26, 2.7: 27, 2.8: 28, 2.9: 29, 3.0: 30, 3.1: 31, 3.2: 32, 3.3: 33, 3.4: 34, 3.5: 35, 3.6: 36, 3.7: 37, 3.8: 38, 3.9: 39, 4.0: 40, 4.1: 41, 4.2: 42, 4.3: 43, 4.4: 44, 4.5: 45, 4.7: 46, 4.8: 47, 5.0: 48, 5.1: 49, 5.2: 50, 5.3: 51, 5.7: 52, 6.6: 53, 7.0: 54, 1684325040.0: 55, 5.5: 56, 5.4: 57, 6.0: 58, 7.1: 59}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 2.9: 1, 4.0: 2, 4.8: 3, 5.8: 4, 9.5: 5, 10.0: 6, 11.0: 7, 12.0: 8, 13.0: 9, 14.0: 10, 15.0: 11, 16.0: 12, 17.0: 13, 18.0: 14, 19.0: 15, 22.0: 16, 23.0: 17, 27.0: 18, 29.0: 19, 31.0: 20, 32.0: 21, 33.0: 22, 36.0: 23, 37.0: 24, 39.0: 25, 40.0: 26, 41.0: 27, 42.0: 28, 43.0: 29, 44.0: 30, 45.0: 31, 46.0: 32, 48.0: 33, 50.0: 34, 51.0: 35, 52.0: 36, 54.0: 37, 55.0: 38, 56.0: 39, 57.0: 40, 58.0: 41, 59.0: 42, 60.0: 43, 61.0: 44, 62.0: 45, 63.0: 46, 64.0: 47, 65.0: 48, 66.0: 49, 67.0: 50, 68.0: 51, 69.0: 52, 70.0: 53, 71.0: 54, 72.0: 55, 73.0: 56, 74.0: 57, 75.0: 58, 76.0: 59, 77.0: 60, 78.0: 61, 79.0: 62, 80.0: 63, 81.0: 64, 82.0: 65, 83.0: 66, 84.0: 67, 85.0: 68, 86.0: 69, 87.0: 70, 88.0: 71, 89.0: 72, 90.0: 73, 91.0: 74, 92.0: 75, 93.0: 76, 94.0: 77, 95.0: 78, 96.0: 79, 97.0: 80, 98.0: 81, 99.0: 82, 100.0: 83, 101.0: 84, 102.0: 85, 103.0: 86, 104.0: 87, 105.0: 88, 106.0: 89, 107.0: 90, 108.0: 91, 109.0: 92, 110.0: 93, 111.0: 94, 112.0: 95, 113.0: 96, 114.0: 97, 115.0: 98, 116.0: 99, 117.0: 100, 118.0: 101, 119.0: 102, 120.0: 103, 121.0: 104, 122.0: 105, 123.0: 106, 124.0: 107, 125.0: 108, 126.0: 109, 127.0: 110, 128.0: 111, 129.0: 112, 130.0: 113, 131.0: 114, 132.0: 115, 133.0: 116, 134.0: 117, 135.0: 118, 136.0: 119, 137.0: 120, 138.0: 121, 139.0: 122, 140.0: 123, 141.0: 124, 142.0: 125, 143.0: 126, 144.0: 127, 145.0: 128, 146.0: 129, 147.0: 130, 148.0: 131, 149.0: 132, 150.0: 133, 151.0: 134, 152.0: 135, 153.0: 136, 154.0: 137, 155.0: 138, 156.0: 139, 157.0: 140, 158.0: 141, 159.0: 142, 160.0: 143, 161.0: 144, 162.0: 145, 163.0: 146, 164.0: 147, 165.0: 148, 166.0: 149, 167.0: 150, 168.0: 151, 169.0: 152, 170.0: 153, 171.0: 154, 172.0: 155, 174.0: 156, 175.0: 157, 176.0: 158, 177.0: 159, 179.0: 160, 180.0: 161, 181.0: 162, 182.0: 163, 183.0: 164, 184.0: 165, 187.0: 166, 189.0: 167, 191.0: 168, 192.0: 169, 193.0: 170, 194.0: 171, 196.0: 172, 198.0: 173, 201.0: 174, 203.0: 175, 204.0: 176, 205.0: 177, 210.0: 178, 212.0: 179, 214.0: 180, 216.0: 181, 220.0: 182, 222.0: 183, 223.0: 184, 225.0: 185, 230.0: 186, 246.0: 187, 248.0: 188, 250.0: 189, 253.0: 190, 256.0: 191, 261.0: 192, 272.0: 193, 273.0: 194, 301.0: 195, 372.0: 196, 1684325040.0: 197, 35.0: 198, 199.0: 199, 213.0: 200, 239.0: 201, 257.0: 202, 235.0: 203, 3.0: 204, 252.0: 205, 178.0: 206, 209.0: 207, 233.0: 208, 197.0: 209, 6.0: 210, 258.0: 211, 430.0: 212, 49.0: 213, 263.0: 214, 219.0: 215, 28.0: 216}, {1993550816.0: 0, 2238339752.0: 1}, {0.36: 0, 0.41: 1, 0.48: 2, 0.49: 3, 0.5: 4, 0.53: 5, 0.54: 6, 0.56: 7, 0.58: 8, 0.59: 9, 0.6: 10, 0.61: 11, 0.62: 12, 0.63: 13, 0.64: 14, 0.65: 15, 0.66: 16, 0.67: 17, 0.68: 18, 0.69: 19, 0.7: 20, 0.71: 21, 0.72: 22, 0.73: 23, 0.74: 24, 0.75: 25, 0.76: 26, 0.77: 27, 0.78: 28, 0.79: 29, 0.8: 30, 0.81: 31, 0.82: 32, 0.83: 33, 0.84: 34, 0.85: 35, 0.86: 36, 0.87: 37, 0.88: 38, 0.89: 39, 0.9: 40, 0.91: 41, 0.92: 42, 0.93: 43, 0.94: 44, 0.944: 45, 0.95: 46, 0.96: 47, 0.97: 48, 0.98: 49, 0.99: 50, 1.0: 51, 1.01: 52, 1.02: 53, 1.03: 54, 1.04: 55, 1.05: 56, 1.06: 57, 1.07: 58, 1.08: 59, 1.09: 60, 1.1: 61, 1.11: 62, 1.12: 63, 1.13: 64, 1.14: 65, 1.15: 66, 1.16: 67, 1.17: 68, 1.18: 69, 1.19: 70, 1.2: 71, 1.21: 72, 1.22: 73, 1.23: 74, 1.24: 75, 1.25: 76, 1.26: 77, 1.27: 78, 1.28: 79, 1.29: 80, 1.3: 81, 1.31: 82, 1.32: 83, 1.33: 84, 1.34: 85, 1.35: 86, 1.36: 87, 1.37: 88, 1.38: 89, 1.39: 90, 1.4: 91, 1.41: 92, 1.42: 93, 1.43: 94, 1.44: 95, 1.45: 96, 1.46: 97, 1.47: 98, 1.49: 99, 1.51: 100, 1.52: 101, 1.53: 102, 1.55: 103, 1.57: 104, 1.58: 105, 1.59: 106, 1.62: 107, 1.65: 108, 1.66: 109, 1.67: 110, 1.68: 111, 1.69: 112, 1.71: 113, 1.74: 114, 1.75: 115, 1.76: 116, 1.77: 117, 1.8: 118, 1.82: 119, 1.83: 120, 2.01: 121, 2.03: 122, 2.32: 123, 1684325040.0: 124, 1.63: 125, 1.48: 126, 1.88: 127, 2.12: 128, 1.7: 129, 1.97: 130, 1.5: 131, 1.73: 132, 1.54: 133, 1.61: 134, 0.25: 135, 0.47: 136, 0.38: 137, 0.46: 138}, {1993550816.0: 0, 2238339752.0: 1}, {2.0: 0, 2.8: 1, 4.0: 2, 5.4: 3, 7.0: 4, 8.4: 5, 8.5: 6, 8.9: 7, 9.0: 8, 10.0: 9, 11.0: 10, 13.0: 11, 14.0: 12, 15.0: 13, 16.0: 14, 17.0: 15, 26.0: 16, 27.0: 17, 28.0: 18, 32.0: 19, 33.0: 20, 34.0: 21, 36.0: 22, 37.0: 23, 39.0: 24, 41.0: 25, 42.0: 26, 46.0: 27, 47.0: 28, 49.0: 29, 50.0: 30, 51.0: 31, 53.0: 32, 54.0: 33, 55.0: 34, 56.0: 35, 57.0: 36, 58.0: 37, 60.0: 38, 61.0: 39, 62.0: 40, 63.0: 41, 64.0: 42, 65.0: 43, 66.0: 44, 67.0: 45, 68.0: 46, 69.0: 47, 70.0: 48, 71.0: 49, 72.0: 50, 73.0: 51, 74.0: 52, 75.0: 53, 76.0: 54, 77.0: 55, 78.0: 56, 79.0: 57, 80.0: 58, 81.0: 59, 82.0: 60, 83.0: 61, 84.0: 62, 85.0: 63, 86.0: 64, 87.0: 65, 88.0: 66, 89.0: 67, 90.0: 68, 91.0: 69, 92.0: 70, 93.0: 71, 94.0: 72, 95.0: 73, 96.0: 74, 97.0: 75, 98.0: 76, 99.0: 77, 100.0: 78, 101.0: 79, 102.0: 80, 103.0: 81, 104.0: 82, 105.0: 83, 106.0: 84, 107.0: 85, 108.0: 86, 109.0: 87, 110.0: 88, 111.0: 89, 112.0: 90, 113.0: 91, 114.0: 92, 115.0: 93, 116.0: 94, 117.0: 95, 118.0: 96, 119.0: 97, 120.0: 98, 121.0: 99, 122.0: 100, 123.0: 101, 124.0: 102, 125.0: 103, 126.0: 104, 127.0: 105, 128.0: 106, 129.0: 107, 130.0: 108, 131.0: 109, 132.0: 110, 133.0: 111, 134.0: 112, 135.0: 113, 136.0: 114, 137.0: 115, 138.0: 116, 139.0: 117, 140.0: 118, 141.0: 119, 142.0: 120, 143.0: 121, 144.0: 122, 145.0: 123, 146.0: 124, 147.0: 125, 148.0: 126, 149.0: 127, 150.0: 128, 151.0: 129, 152.0: 130, 153.0: 131, 154.0: 132, 155.0: 133, 156.0: 134, 157.0: 135, 158.0: 136, 159.0: 137, 160.0: 138, 161.0: 139, 162.0: 140, 164.0: 141, 165.0: 142, 166.0: 143, 167.0: 144, 168.0: 145, 169.0: 146, 170.0: 147, 171.0: 148, 172.0: 149, 173.0: 150, 174.0: 151, 175.0: 152, 176.0: 153, 177.0: 154, 178.0: 155, 179.0: 156, 180.0: 157, 181.0: 158, 183.0: 159, 185.0: 160, 186.0: 161, 187.0: 162, 188.0: 163, 194.0: 164, 195.0: 165, 197.0: 166, 198.0: 167, 199.0: 168, 200.0: 169, 201.0: 170, 204.0: 171, 205.0: 172, 207.0: 173, 213.0: 174, 215.0: 175, 217.0: 176, 220.0: 177, 222.0: 178, 224.0: 179, 232.0: 180, 235.0: 181, 247.0: 182, 280.0: 183, 281.0: 184, 283.0: 185, 291.0: 186, 1684325040.0: 187, 189.0: 188, 190.0: 189, 209.0: 190, 59.0: 191, 203.0: 192, 214.0: 193, 3.0: 194, 242.0: 195, 206.0: 196, 253.0: 197, 40.0: 198, 163.0: 199, 7.6: 200, 227.0: 201, 312.0: 202, 395.0: 203, 265.0: 204, 274.0: 205, 9.1: 206, 19.0: 207, 196.0: 208, 210.0: 209, 219.0: 210}, {1993550816.0: 0}, {1684325040.0: 0}, {596708387.0: 0, 1203304565.0: 1, 1918519837.0: 2, 3646436640.0: 3, 3655101910.0: 4}]
target = ''
target_column = 29
important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
ignore_idxs = []
classifier_type = 'NN'
num_attr = 29
n_classes = 2
model_cap = 53
w_h = np.array([[0.08141525089740753, 0.36700519919395447, -0.5087605118751526, -0.0956449881196022, 0.4388064742088318, -0.17613494396209717, -0.47851720452308655, 0.5859803557395935, 1.0312856435775757, 0.5305870771408081, -0.4290519654750824], [-0.21829351782798767, -0.6380777359008789, -0.20873276889324188, -0.010284814983606339, 0.0437692292034626, -0.04735143110156059, 0.1427372694015503, -0.4307993948459625, -0.6887665390968323, 0.3820556104183197, -0.9401283264160156], [0.011352291330695152, 0.18999071419239044, -0.08527600020170212, -0.09176398813724518, 0.2012120485305786, -0.40405112504959106, 0.1197272390127182, 0.4534895420074463, -1.9995449781417847, -0.2637369632720947, -0.029101962223649025], [-0.26544132828712463, -0.6614785194396973, -0.012827130034565926, -0.11481011658906937, 0.16787658631801605, -0.5336595177650452, 0.37821632623672485, 0.40517374873161316, -0.15531937777996063, 0.38849371671676636, 0.6553348898887634]])
b_h = np.array([-3.2984204292297363, 0.9511513710021973, -2.3196001052856445, -2.63155198097229])
w_o = np.array([[-0.28358784317970276, -0.613957405090332, 0.5092752575874329, 0.617421567440033]])
b_o = np.array(-4.757114887237549)


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

def __transform(X):
    mean = np.array([51.621007038440716, 0.6556578234975636, 0.12669193286410396, 0.014618299945858148, 0.01028695181375203, 0.03898213318895506, 0.01245262587980509, 0.012994044396318355, 0.017325392528424473, 0.061180292365998916, 0.06334596643205198, 0.005414185165132647, 0.007038440714672442, 0.02707092582566324, 0.0005414185165132648, 0.04981050351922036, 0.8949648077964266, 107.43584190579318, 0.797509474824039, 26.790471034109366, 0.9334055224688684, 97.25392528424472, 0.8857606930157012, 58.63616675690309, 0.8863021115322144, 98.32918245804007, 0.0, 0.0, 2.4612885760693017])
    components = np.array([[-0.03989265973600438, -0.000851898702087489, -0.00033261945961156293, 5.2966719560881735e-06, -7.239305550372038e-05, -4.560194116598192e-05, -3.4273915721167726e-05, -7.148075366786019e-05, -9.05075298944606e-05, 5.7281484332038596e-05, -1.1078486744810857e-05, 4.544033726306191e-06, 8.515528557347893e-05, 9.144864262569525e-05, -1.0761523933794557e-05, -8.091861529649758e-05, -0.003309848244589321, 0.9386538156641295, -0.0021486165609143153, 0.0718830120879374, -0.0022761264181517292, 0.18970896226183376, -0.0025129962847226486, 0.216163676531875, -0.002521373743903544, 0.17151334463188248, 0.0, 0.0, 0.0017495830547092166], [-0.01380610265181991, -0.002251485820385847, 0.0009318463901971377, -4.20513976478754e-05, 5.6123792934348266e-05, -8.50033343274781e-05, 0.0001267940139767566, 3.913952013469141e-05, -6.937151102617598e-05, -0.00018094825886014808, 0.00041934768294195546, -4.64396095704234e-05, -6.642649262416915e-05, 0.00015332421497830394, -9.811847714900344e-06, -0.00015965349402409504, -0.001195582919127835, -0.31841745072624683, -0.0015765098849260463, 0.09753693609194117, -0.002135859223602459, 0.6220900380595573, -0.0031452537022812355, 0.2880475815282576, -0.0031466948189448002, 0.6472264381385016, -0.0, -0.0, 0.0015067990621467104], [0.5242897868147571, 0.006376139695304544, -0.0005286333905504043, 6.15829749250192e-05, -0.00031313112471470256, 0.0005965605120856766, -0.0012690482759614063, -8.101279951359024e-05, -1.577919634916317e-05, 6.392830653555675e-05, -0.0008188910934456155, -3.435752517804443e-05, -0.00018405538396519322, -0.00040615654997168394, -4.306109174062413e-05, 5.938550810631812e-06, -0.00020413777893471085, 0.09481504932371476, 0.001887743770393607, -0.1960848038638505, 0.0004793947871614893, -0.34398575798446884, -0.0021479565178677785, -0.4321683218948547, -0.002181603955571429, 0.6103448653015626, -0.0, -0.0, 0.0027324745632318174], [0.5021663544164378, -0.004047460644791257, 0.0014660291471313455, 5.1441602686121764e-05, -0.00027888746712326164, 0.00042725203759869486, 6.838075466100358e-06, -0.0002478251961125054, 8.458327650900943e-05, -0.00011064119321331395, 0.00022289612855394333, 1.5603889020296253e-05, -1.0777213735038918e-05, 0.0001345469939386882, -4.110461155336281e-05, -0.00012734769317998004, -0.0018878121299450465, 0.05872538382575768, -0.0035888879167869465, 0.14315945663130222, -0.0010752174876864638, 0.6153289269502464, 0.006897451622438725, -0.45595747892268024, 0.006797921875057767, -0.3704418205261412, -0.0, -0.0, -0.0011183725591527912], [0.6675452696601883, -0.00480371443420556, -0.0016408324637931218, -0.0002506725026936238, -7.854816384648032e-05, 0.0003252880052079163, 0.0004408243341178153, -5.093371313936388e-05, 0.000256488578752721, 0.0004971737278294266, -9.418213123924414e-05, -9.727548684353741e-05, -0.00016543011953300022, -7.768362884164103e-05, 8.01589135192226e-06, -0.0013132851148141634, 0.0019053928049920924, -0.05986181363304628, 0.004479782807344166, -0.1739847831285202, -0.000423206933539699, -0.10572535221200535, -0.0037043303881000213, 0.6869953839113933, -0.003638861288132789, -0.1931566596080924, -0.0, -0.0, 0.00031307071274056165], [0.15978914210693917, -0.0005809908602427008, 0.003084861443987367, 0.000283112168105413, 3.7866637362207946e-05, -0.0009137989099047487, -0.00021919220524839003, 0.00032216665791895367, 0.0005598075802188057, 0.0006409448082383788, -0.0004240478101105329, 0.0001462200252105204, 0.0003243253830949309, 5.2952802454552436e-05, -1.4465244134075121e-05, -0.0014289829627595065, -0.0025466359031983657, -0.038770126475309606, -0.023102901962314074, 0.9462402649484144, -0.0003737225464994105, -0.2622935671364162, -0.001992949462073017, 0.05964966869197169, -0.0019819378627456935, 0.06717864000025493, 0.0, 0.0, 0.0127449345368345], [0.0029907814482658477, 0.9205302536614154, -0.05094396702026413, 0.005085871714516394, -0.002324974938206806, -0.0035050669944242763, 0.0019176501401686088, -0.004529963121548491, -0.009498902012439693, -0.013317907842777242, -0.017670973504280478, 0.0010754382875553395, 0.0019471784798789217, -0.008682981661390095, 0.0007951047931883333, 0.0552896768537136, -0.009371566383118502, -0.0003115030340590934, 0.013454355806274739, 0.006628922423390623, -0.01211104861476779, 0.003749413451770363, -0.008947629496302978, 0.004974459893063249, -0.008035856703985333, -0.0028266465402248075, 0.0, 0.0, -0.38155378539248025], [0.00216303568405396, -0.38726524932648276, -0.056342922013189724, -0.009881835004667197, -0.004355120030185911, -0.019250999049047573, 0.013962972454185454, -0.007818775674055518, -0.013058678895786293, -0.0027378829103553255, -0.021347356346966436, 0.012224901103213174, -0.002927253542802939, 0.009092999018841698, -0.00046303898691565394, 0.0883346093794565, 0.03511585878798044, 0.0011151236579286766, 0.012985186603829585, 0.010630380641271238, 0.01670033934917741, -0.005538164894343486, 0.012482211614167798, -0.0008177552091255915, 0.012283466720329395, 0.005840224167285442, -0.0, -0.0, -0.9139247390933668], [0.00026507051780574263, 0.02969320008486543, 0.9622921453798896, 0.002895011907612003, -0.010884002148787209, -0.02451219834968582, 0.008771666805538971, 0.016752024071489218, -0.0012294588154853755, 0.18330728493325923, -0.11185118080794164, 0.0018684228507944, 0.006251094649924743, -0.03841982343397624, -0.0003268261682823626, 0.013878451080735127, 0.09338088827463589, 0.0009567542250724151, -0.06104546062402485, -0.0038347145870301345, 0.06804778714224712, -0.0006579895300068529, 0.039171307130645934, 0.00190432064702427, 0.039473468986831906, 0.00046714947584901816, -0.0, -0.0, -0.06334190150211545], [0.0004914212613613948, 0.016689708147354643, -0.028871541428792332, -0.016171624193592447, 0.02567897323901592, 0.00466567845653554, 0.03640638683536167, 0.0024911258931430156, 0.03354712740383921, 0.5749053825172967, 0.7906832267414469, 0.002460527758229157, -0.011716135311560223, 0.02197127252988507, -0.0017432987484339761, 0.00387302191341232, 0.058553716351070796, 0.0006655488713693638, 0.12408905827920198, 0.003462779682884599, 0.09941634985880217, -0.0006547007509351509, 0.07184881247790006, 0.0003431096609437871, 0.07050594385322993, 0.001499055439819197, -0.0, -0.0, -0.016995116898611776], [-0.0007123111267148296, 0.0008960053437785238, -0.23204288003747467, -0.028866400468475727, -0.03240482328844696, 0.14389605504169242, -0.046759348466775104, -0.0007260680813224949, -0.0346935252058377, 0.7323078662233955, -0.5530223874331085, -0.008339339478017294, -0.011639939026560854, -0.06045038631510682, -0.0012327590454763767, 0.14109167768637615, 0.2087877550457544, 0.0003055594751008648, -0.06515936553994406, -0.0005476081461135407, 0.05558465519996432, 0.0015505815677779957, 0.027600002292035807, 9.767109381574794e-05, 0.026452045556123592, 0.00022818011705754126, 0.0, 0.0, 0.043663282065554705]])
    explained_variance = np.array([4909.1411973947925, 3145.5960779255547, 512.9029478716407, 448.1874056712116, 393.28825840863215, 171.42186145083392, 0.8317969300285151, 0.544135647370929, 0.1040687549949058, 0.06151764780766557, 0.05412701216092067])
    X = X - mean
    X_transformed = np.dot(X, components.T)
    return X_transformed


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

    arr = __transform(arr)

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
