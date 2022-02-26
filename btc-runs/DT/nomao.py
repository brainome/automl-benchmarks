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
# Invocation: brainome TRAIN_TEST_SPLITS/phpDYCOet-clean-train.csv -f DT -y -split 70 -modelonly -q -o btc-runs/DT/nomao.py -json btc-runs/DT/nomao.json -rank
# Total compiler execution time: 0:00:40.88. Finished on: Feb-26-2022 19:07:14.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/DT/nomao.py
    Classifier Type:              Decision Tree
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        71.44%
      Training accuracy:          96.44% (16285/16886 correct)
      Validation Accuracy:        91.98% (6659/7239 correct)
      Combined Model Accuracy:    95.10% (22944/24125 correct)


    Model Capacity (MEC):        789    bits
    Generalization Ratio:         17.81 bits/bit
    Percent of Data Memorized:    11.87%
    Resilience to Noise:          -1.31 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   1 |   4475    348 
                   0 |    253  11810 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   1 |   1731    337 
                   0 |    243   4928 

    Training Accuracy by Class:
               Class |     TP     FP     TN     FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ------ ------ ------ ------ -------- -------- -------- -------- -------- --------
                   1 |   4475    253  11810    348   92.78%   97.90%   94.65%   97.14%   93.71%   88.16%
                   0 |  11810    348   4475    253   97.90%   92.78%   97.14%   94.65%   97.52%   95.16%

    Validation Accuracy by Class:
               Class |     TP     FP     TN     FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ------ ------ ------ ------ -------- -------- -------- -------- -------- --------
                   1 |   1731    243   4928    337   83.70%   95.30%   87.69%   93.60%   85.65%   74.90%
                   0 |   4928    337   1731    243   95.30%   83.70%   93.60%   87.69%   94.44%   89.47%




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
TRAINFILE = ['TRAIN_TEST_SPLITS/phpDYCOet-clean-train.csv']
mapping = {'1': 0, '0': 1}
ignorelabels = []
ignorecolumns = ['V1', 'V2', 'V3', 'V5', 'V6', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V92', 'V93', 'V94', 'V95', 'V96', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118']
target = ''
target_column = 118
important_idxs = [3, 6, 7, 90, 96]
ignore_idxs = [0, 1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
classifier_type = 'DT'
num_attr = 118
n_classes = 2
model_cap = 789
energy_thresholds = [2.1243895000000004, 2.132344, 2.3627615, 2.363174, 2.626473, 2.633204, 2.643147, 2.6467180000000003, 2.6670430000000005, 2.6676865000000003, 2.720571, 2.7211125, 2.7789595, 2.779629, 2.783153, 2.7833275, 2.8494805000000003, 2.850658, 2.9150725, 2.9152044999999998, 2.9731180000000004, 2.973472, 2.999768, 3.0003625, 3.0064520000000003, 3.0068799999999998, 3.014432, 3.0145315, 3.0275245, 3.0288765, 3.0293115, 3.0297435000000004, 3.034852, 3.034957, 3.0489165000000003, 3.0491955, 3.0555725000000002, 3.055602, 3.0599365, 3.0606774999999997, 3.0728655, 3.073416, 3.083123, 3.083567, 3.0903105, 3.091144, 3.0998985, 3.1007375, 3.1014929999999996, 3.10174, 3.1162520000000002, 3.1169345, 3.1195535000000003, 3.1200925, 3.1283075, 3.128781, 3.129905, 3.130459, 3.1412925, 3.141967, 3.1433545, 3.1435215000000003, 3.144285, 3.1448475, 3.1567380000000003, 3.156866, 3.159456, 3.1609295, 3.1625615, 3.1636075000000003, 3.1664605, 3.1666670000000003, 3.1675294999999997, 3.1678319999999998, 3.1796585, 3.1798975, 3.1856365, 3.186315, 3.1903435, 3.19083, 3.199199, 3.1994495, 3.2021735, 3.2024695, 3.2029955, 3.2034640000000003, 3.2079495, 3.2090365, 3.211702, 3.211814, 3.222263, 3.2226035, 3.2319515, 3.2326034999999997, 3.2497585, 3.2504644999999996, 3.2517295, 3.251961, 3.2521295000000006, 3.252576, 3.256217, 3.256622, 3.258092, 3.258419, 3.260954, 3.2616050000000003, 3.2624224999999996, 3.263802, 3.2717845, 3.2731665000000003, 3.2829325, 3.2835955, 3.285582, 3.2857975, 3.286323, 3.2866495000000002, 3.2926685, 3.2935730000000003, 3.2940329999999998, 3.294101, 3.3071915, 3.3080654999999997, 3.3082395, 3.3089565, 3.3120214999999997, 3.31316, 3.324517, 3.324846, 3.325856, 3.3267175, 3.3322849999999997, 3.3336734999999997, 3.3386175, 3.3393215, 3.342531, 3.3439395000000003, 3.3480585, 3.34835, 3.3485715000000003, 3.348868, 3.3509525, 3.3515535, 3.3639945, 3.3648439999999997, 3.3673184999999997, 3.368318, 3.368734, 3.3691825, 3.3719245, 3.3729825, 3.37584, 3.377186, 3.3821494999999997, 3.3829115, 3.3840705, 3.384732, 3.3939779999999997, 3.395091, 3.4008255, 3.4016145, 3.413988, 3.4145700000000003, 3.4153205, 3.4160779999999997, 3.4164250000000003, 3.4166665, 3.417593, 3.418577, 3.425267, 3.425917, 3.4278205, 3.428925, 3.4406280000000002, 3.440864, 3.442735, 3.4441695, 3.44562, 3.4481599999999997, 3.4497004999999996, 3.4503415, 3.4523099999999998, 3.4528355, 3.455844, 3.4570295, 3.4579959999999996, 3.4584455, 3.460651, 3.4625565, 3.46273, 3.4637, 3.475459, 3.4763525, 3.483069, 3.4833335, 3.4855080000000003, 3.486253, 3.48871, 3.488823, 3.489899, 3.491288, 3.4937975000000003, 3.49871, 3.4997585, 3.5021775, 3.5058535, 3.5066785, 3.5149215, 3.5171989999999997, 3.5191654999999997, 3.5219635, 3.5257555, 3.527587, 3.5323884999999997, 3.5338375, 3.535298, 3.535998, 3.5420204999999996, 3.543211, 3.545847, 3.548441, 3.5507605, 3.551419, 3.5519524999999996, 3.552468, 3.5530885, 3.5538194999999995, 3.55666, 3.558004, 3.5622585, 3.5639114999999997, 3.5672629999999996, 3.567589, 3.568684, 3.5687835, 3.5727045, 3.5731979999999997, 3.5748935, 3.5751735, 3.5770210000000002, 3.577642, 3.5830355, 3.5834325, 3.5878015000000003, 3.5881205, 3.5891329999999995, 3.5892729999999995, 3.5896575, 3.589964, 3.590783, 3.5910944999999996, 3.591572, 3.591744, 3.593914, 3.5940819999999998, 3.5953209999999998, 3.5954534999999996, 3.595783, 3.5961355, 3.5963765, 3.5964929999999997, 3.5976619999999997, 3.5980214999999998, 3.598566, 3.5987669999999996, 3.5992819999999996, 3.599344, 3.5999999999999996, 3.6001054999999997, 3.6029514999999996, 3.6029935, 3.6051504999999997, 3.6052595, 3.6069069999999996, 3.6070149999999996, 3.6095635, 3.6100825, 3.6113869999999997, 3.6116789999999996, 3.620682, 3.6210484999999997, 3.621508, 3.6216045, 3.626772, 3.6269885000000004, 3.631158, 3.6314915, 3.6440805000000003, 3.6444444999999996, 3.659974, 3.6604900000000002, 3.665465, 3.6661345, 3.6684955, 3.6693610000000003, 3.6707345, 3.671462, 3.671678, 3.672631, 3.6735305, 3.6742805, 3.6767425, 3.6783045000000003, 3.6816069999999996, 3.6831544999999997, 3.6846134999999998, 3.685542, 3.691369, 3.693892, 3.6965755, 3.6972709999999998, 3.699811, 3.700099, 3.7011795, 3.7025555, 3.7065775, 3.7084265, 3.7130840000000003, 3.7147915, 3.722295, 3.7223945, 3.7226619999999997, 3.7240105000000003, 3.7254955, 3.727153, 3.729873, 3.7310455, 3.7318834999999995, 3.733815, 3.7347159999999997, 3.736643, 3.7432109999999996, 3.744104, 3.7524524999999995, 3.7543175, 3.761107, 3.762259, 3.7628854999999994, 3.7633909999999995, 3.7640969999999996, 3.765379, 3.7713815000000004, 3.7726825, 3.7749085, 3.7766089999999997, 3.77954, 3.782132, 3.783587, 3.7840629999999997, 3.7855585, 3.7857339999999997, 3.7865355, 3.78667, 3.787335, 3.788018, 3.788485, 3.7888384999999998, 3.791035, 3.7912105, 3.791629, 3.7923845, 3.79405, 3.7943655, 3.7960865, 3.796342, 3.7980405, 3.7981805, 3.7983969999999996, 3.7987865, 3.8010554999999995, 3.8034584999999996, 3.806151, 3.8084515000000003, 3.8088195000000002, 3.809368, 3.8105539999999998, 3.8114075000000005, 3.811437, 3.811619, 3.8117055, 3.8122784999999997, 3.81247, 3.8125904999999998, 3.8137919999999994, 3.8147915, 3.8151085, 3.815528, 3.8160714999999996, 3.818646, 3.8199915, 3.8209194999999996, 3.821506, 3.8223215, 3.8224275, 3.8229300000000004, 3.8238525, 3.824527, 3.824873, 3.8256265, 3.8266345, 3.826696, 3.826804, 3.827428, 3.8280895, 3.8282969999999996, 3.829409, 3.8306505, 3.831074, 3.83378, 3.835033, 3.8358135, 3.836319, 3.8367155, 3.83704, 3.8389879999999996, 3.839879, 3.8413259999999996, 3.8414805, 3.8446755, 3.8467015, 3.8473214999999996, 3.848115, 3.849989, 3.850016, 3.8503425, 3.8513705, 3.8518255, 3.8526365, 3.853709, 3.8546065, 3.8556165, 3.8571785, 3.8585339999999997, 3.8597209999999995, 3.8601599999999996, 3.8619795000000003, 3.8630295, 3.864782, 3.8651359999999997, 3.865321, 3.8654555000000004, 3.8655635000000004, 3.8667555, 3.8668975000000003, 3.8677745, 3.868912, 3.869964, 3.870053, 3.870777, 3.8741855, 3.875041, 3.875928, 3.8775205, 3.8783905, 3.8786585000000002, 3.879075, 3.880853, 3.881672, 3.8858854999999997, 3.8865369999999997, 3.888052, 3.8892265, 3.8898580000000003, 3.8905275, 3.892177, 3.8938555, 3.899825, 3.900566, 3.9035195, 3.903772, 3.9046425, 3.9059904999999997, 3.906429, 3.907029, 3.9075165, 3.9115374999999997, 3.9140289999999998, 3.9151784999999997, 3.915789, 3.9161835, 3.9166415, 3.9198844999999998, 3.9218025, 3.923248, 3.9251829999999996, 3.9273654999999996, 3.929385, 3.929658, 3.9302894999999998, 3.9322850000000003, 3.9339440000000003, 3.9425749999999997, 3.9449585, 3.9461315, 3.947939, 3.9492744999999996, 3.9519725, 3.9524980000000003, 3.9554, 3.9562409999999995, 3.9594245, 3.9605819999999996, 3.961954, 3.9646005, 3.9678269999999998, 3.9727115, 3.9734689999999997, 3.9775460000000002, 3.978332, 3.9795404999999997, 3.9800554999999997, 3.983373, 3.9864435, 3.9866574999999997, 3.9870305, 3.9893285, 3.990539, 3.996247, 4.0000990000000005, 4.000489, 4.000923, 4.0044375, 4.005612, 4.006832, 4.014791499999999, 4.016121, 4.017193000000001, 4.01845, 4.020943, 4.0215155000000005, 4.0239235, 4.023978, 4.026243000000001, 4.026549, 4.0292829999999995, 4.0293495, 4.035734, 4.0363185, 4.0370595, 4.0375075, 4.040224, 4.040709, 4.042201, 4.043991, 4.0478404999999995, 4.048816, 4.0500039999999995, 4.050146, 4.0572615, 4.058976, 4.066765999999999, 4.0670459999999995, 4.0723415, 4.072574, 4.073169, 4.073942000000001, 4.074918, 4.0756215000000005, 4.0764225, 4.0767425, 4.0776895, 4.079344, 4.0804765, 4.081391999999999, 4.085041, 4.0851755, 4.0854495, 4.0891865, 4.0913260000000005, 4.091536, 4.0938375, 4.0948815, 4.098257, 4.0993055, 4.108929, 4.110384, 4.1134260000000005, 4.113866, 4.114432, 4.114928, 4.1221155, 4.124226500000001, 4.13006, 4.130352, 4.1325994999999995, 4.133005000000001, 4.133555, 4.1340215, 4.1376045, 4.138871, 4.144067, 4.1451955, 4.1486545, 4.149173, 4.1501585, 4.150359, 4.1507755, 4.1510765, 4.156506, 4.157417000000001, 4.167333, 4.1688485, 4.1738029999999995, 4.177365, 4.1830715000000005, 4.1844475, 4.2085740000000005, 4.208914500000001, 4.2245435, 4.224728499999999, 4.2368025, 4.237266, 4.2410615, 4.241766, 4.2529105, 4.254308999999999, 4.260547000000001, 4.262521, 4.2637465, 4.2643284999999995, 4.271378, 4.2725685, 4.2815905, 4.2827269999999995, 4.286512500000001, 4.2873985, 4.2882935, 4.289942, 4.300142, 4.301177, 4.3379035, 4.338887, 4.364103, 4.3645700000000005, 4.364574, 4.364928, 4.366524500000001, 4.3675595000000005, 4.3814235, 4.3860725, 4.386393, 4.386607, 4.4018, 4.403618, 4.432255499999999, 4.434794, 4.455553, 4.455690499999999, 4.4598315, 4.460824, 4.47877, 4.478856, 4.5053905, 4.5056495000000005, 4.507816500000001, 4.5107845, 4.5257435, 4.5261405, 4.5312434999999995, 4.5317405, 4.548189499999999, 4.549522, 4.5537155, 4.556583, 4.5645785, 4.565626999999999, 4.5940460000000005, 4.595489000000001, 4.600407499999999, 4.601157499999999, 4.6091725, 4.610908499999999, 4.6155655, 4.617430499999999, 4.6196205, 4.621271, 4.640377, 4.6420205, 4.667223, 4.6685490000000005, 4.706830500000001, 4.709580000000001, 4.720772500000001, 4.7221115000000005, 4.7408305, 4.7467825, 4.7625315, 4.763864, 4.8144, 4.8146925, 4.8290135, 4.833092000000001, 4.852171, 4.855275499999999, 4.86408, 4.8661069999999995, 4.922874, 4.927965, 4.960969, 4.9664255, 5.1129785000000005, 5.1175735, 5.149586, 5.159183500000001, 5.168903, 5.171284, 5.209269, 5.2112605, 5.2327379999999994, 5.2344805, 5.2843705, 5.2857145, 5.292796, 5.296155499999999, 5.3160035, 5.3232495, 5.374916, 5.378367, 5.4147815, 5.4232320000000005, 5.435014499999999, 5.441618, 5.4676485, 5.474641500000001, 5.506579, 5.513864, 5.5500385, 5.559596, 5.593837499999999, 5.5971969999999995, 5.5989515, 5.616739, 5.765599999999999, 5.766246499999999, 5.767322500000001, 5.768809, 5.7692815, 5.773308500000001, 5.7747245, 5.775101, 5.7757435, 5.776070499999999, 5.7787895, 5.780097, 5.7873090000000005, 5.790483500000001, 5.7935615, 5.796991, 5.802427, 5.80957, 5.818982, 5.832285, 5.840148, 5.8458594999999995, 5.851691, 5.856311, 5.8750990000000005, 5.878887000000001, 5.8892295, 5.8984985000000005, 5.927157, 5.932639, 5.941560000000001, 5.968786, 5.982093000000001, 5.9955985, 6.007785500000001, 6.010317499999999, 6.012211499999999, 6.019238, 6.020008, 6.0281530000000005, 6.0293455, 6.030835, 6.0327955, 6.035185, 6.0377495, 6.04097, 6.045548500000001, 6.0509805, 6.05719, 6.0881055, 6.1055554999999995, 6.133687, 6.162039500000001, 6.166425500000001, 6.2014745, 6.2125855, 6.277877, 6.295592]
start_label = 1


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


def __classify(rows):
    try:
        energys = np.sum(rows, axis=1, dtype=np.float128)
    except:
        energys = np.sum(rows, axis=1, dtype=np.longdouble)
    numers = np.searchsorted(energy_thresholds, energys, side='left') - 1
    indys = np.argwhere(np.logical_and(numers <= len(energy_thresholds), numers >= 0)).reshape(-1)
    defaultindys = np.argwhere(np.logical_not(np.logical_and(numers <= len(energy_thresholds), numers >= 0))).reshape(-1)
    output = np.zeros(energys.shape[0])
    output[indys] = (numers[indys] + 1) % 2
    if list(defaultindys):
        output[defaultindys] = 0
    return output



def __validate_kwargs(kwargs):
    for key in kwargs:

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
        If True, remaps the output to the original class label.

    Returns
    -------
    output : np.ndarray

        A numpy array of predictions.

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
            print("Classifier Type:                    Decision Tree")
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
