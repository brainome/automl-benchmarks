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
# Invocation: brainome TRAIN_TEST_SPLITS/speeddating-clean-train.csv -f DT -y -split 70 -modelonly -q -o btc-runs/DT/SpeedDating.py -json btc-runs/DT/SpeedDating.json -rank
# Total compiler execution time: 0:00:11.43. Finished on: Feb-26-2022 19:08:16.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/DT/SpeedDating.py
    Classifier Type:              Decision Tree
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        83.53%
      Training accuracy:         100.00% (4104/4104 correct)
      Validation Accuracy:        98.29% (1730/1760 correct)
      Combined Model Accuracy:    99.48% (5834/5864 correct)


    Model Capacity (MEC):        347    bits
    Generalization Ratio:          7.63 bits/bit
    Percent of Data Memorized:    30.74%
    Resilience to Noise:          -1.07 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |  3428     0 
                   1 |     0   676 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |  1468     2 
                   1 |    28   262 

    Training Accuracy by Class:
               match |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |  3428     0   676     0  100.00%  100.00%  100.00%  100.00%  100.00%  100.00%
                   1 |   676     0  3428     0  100.00%  100.00%  100.00%  100.00%  100.00%  100.00%

    Validation Accuracy by Class:
               match |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |  1468    28   262     2   99.86%   90.34%   98.13%   99.24%   98.99%   98.00%
                   1 |   262     2  1468    28   90.34%   99.86%   99.24%   98.13%   94.58%   89.73%




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
TRAINFILE = ['TRAIN_TEST_SPLITS/speeddating-clean-train.csv']
mapping = {'0': 0, '1': 1}
ignorelabels = []
ignorecolumns = ['has_null', 'wave', 'gender', 'age', 'age_o', 'd_age', 'd_d_age', 'race', 'race_o', 'samerace', 'importance_same_race', 'importance_same_religion', 'd_importance_same_race', 'd_importance_same_religion', 'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests', 'd_pref_o_attractive', 'd_pref_o_sincere', 'd_pref_o_intelligence', 'd_pref_o_funny', 'd_pref_o_ambitious', 'd_pref_o_shared_interests', 'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o', 'shared_interests_o', 'd_attractive_o', 'd_sinsere_o', 'd_intelligence_o', 'd_funny_o', 'd_ambitous_o', 'd_shared_interests_o', 'attractive_important', 'sincere_important', 'intellicence_important', 'funny_important', 'ambtition_important', 'shared_interests_important', 'd_attractive_important', 'd_sincere_important', 'd_intellicence_important', 'd_funny_important', 'd_ambtition_important', 'd_shared_interests_important', 'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'd_attractive', 'd_sincere', 'd_intelligence', 'd_funny', 'd_ambition', 'attractive_partner', 'sincere_partner', 'intelligence_partner', 'funny_partner', 'ambition_partner', 'shared_interests_partner', 'd_attractive_partner', 'd_sincere_partner', 'd_intelligence_partner', 'd_funny_partner', 'd_ambition_partner', 'd_shared_interests_partner', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'd_sports', 'd_tvsports', 'd_exercise', 'd_dining', 'd_museums', 'd_art', 'd_hiking', 'd_gaming', 'd_clubbing', 'd_reading', 'd_tv', 'd_theater', 'd_movies', 'd_concerts', 'd_music', 'd_shopping', 'd_yoga', 'interests_correlate', 'd_interests_correlate', 'expected_happy_with_sd_people', 'expected_num_interested_in_me', 'expected_num_matches', 'd_expected_happy_with_sd_people', 'd_expected_num_interested_in_me', 'd_expected_num_matches', 'like', 'guess_prob_liked', 'd_like', 'd_guess_prob_liked', 'met']
target = ''
target_column = 122
important_idxs = [14, 120, 121]
ignore_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
classifier_type = 'DT'
num_attr = 122
n_classes = 2
model_cap = 347
energy_thresholds = [15017932.5, 20807076.5, 26596221.5, 35451559.0, 44306897.5, 45106082.0, 45905267.5, 46817832.5, 47730398.5, 51097454.0, 54464510.5, 62481985.0, 148074041.5, 150244566.5, 152415092.5, 152530331.5, 152645571.5, 163360405.5, 174075240.5, 179782980.5, 185490721.5, 193260161.5, 201029601.5, 201549448.5, 202069296.5, 202493697.5, 225501281.5, 234076680.0, 242652079.5, 275006759.5, 307361440.5, 314274577.0, 376497100.5, 385802534.0, 395107968.5, 405734333.0, 435878293.5, 443399446.0, 450920599.5, 455862832.0, 460805065.5, 461887732.0, 465879843.5, 471381789.5, 476883736.5, 481396473.0, 497974145.5, 518514371.5, 539054597.5, 543435043.5, 556358681.5, 561842302.5, 567325924.5, 572744448.5, 578162973.5, 578626042.0, 579089111.5, 603850899.0, 647408933.5, 657984946.0, 677667345.5, 694119657.5, 710571970.5, 717227447.0, 723882924.5, 733999540.5, 744116157.5, 753622210.5, 763128264.5, 773132168.5, 783136073.5, 788057840.0, 792979607.5, 798078246.5, 835505083.5, 856256453.5, 877007824.5, 884767505.5, 892527187.5, 902319341.5, 912111496.5, 913910316.0, 915709136.5, 916742514.0, 919988818.5, 936496787.0, 974512776.5, 981752851.0, 997845352.5, 1023722280.0, 1131395654.5, 1133641579.0, 1135887504.5, 1143336904.5, 1150786305.5, 1189978441.5, 1230742933.5, 1241784671.5, 1252826410.5, 1258465277.5, 1264104145.5, 1265390632.0, 1266677119.5, 1272260280.5, 1277843442.5, 1289892539.5, 1383944572.5, 1398001866.5, 1412059161.5, 1428174671.0, 1444290180.5, 1458607515.0, 1472924850.5, 1476434848.0, 1479944846.5, 1490781308.0, 1501617770.5, 1505647982.0, 1509678194.5, 1511933522.0, 1565826304.5, 1582887185.5, 1615653515.5, 1649989278.0, 1684325041.5, 1694754587.5, 1705543802.5, 1714305829.5, 1732187886.5, 1733065394.0, 1733942902.5, 1735631312.0, 1737319722.5, 1741118624.0, 1822983427.5, 1839178016.0, 1855372605.5, 1866418504.5, 1877464403.5, 1890378396.0, 1903292388.5, 1904687166.0, 1906081944.5, 1935815976.5, 1970028623.5, 1970686649.0, 2021470579.5, 2027176168.0, 2046499944.5, 2049660799.5, 2129020950.5, 2133081006.0, 2137141062.5, 2139053124.0, 2140965186.5, 2141362141.5, 2149714802.5, 2164411973.5, 2179109144.5, 2182326173.5, 2222549893.5, 2226658442.0, 2227358947.5, 2232111210.5, 2236863474.5, 2237040921.5, 2237218369.5, 2255940629.0, 2274662889.5, 2281935785.5, 2289208682.5, 2290519037.5, 2291829393.5, 2294436741.5, 2297044090.5, 2300930721.5, 2304817353.5, 2312618663.0, 2320419973.5, 2338135583.5, 2374151409.5, 2382758145.0, 2428110461.5, 2456176925.5, 2484243390.0, 2485401935.5, 2570404226.5, 2582686868.0, 2594969510.5, 2600399013.0, 2614965781.5, 2622314840.0, 2629663899.5, 2644945134.0, 2660226369.5, 2670463679.5, 2680700990.5, 2689362004.0, 2705344732.5, 2711928015.0, 2718511298.5, 2748219456.5, 2784562540.5, 2787489128.5, 2790415717.5, 2796609627.5, 2837906510.5, 2860444675.5, 2882982841.5, 2887461994.0, 2897068475.5, 2904434007.0, 2911799539.5, 2916590333.0, 2921381127.5, 2921813700.0, 2922246273.5, 2925345638.5, 2929283936.5, 2936508639.5, 2943733343.5, 2945174677.5, 2946616012.5, 2946684544.0, 3007168438.5, 3010237946.5, 3035916903.5, 3040134153.5, 3044351404.5, 3047232751.5, 3050114099.5, 3052353945.0, 3054593791.5, 3054875597.0, 3055157403.5, 3068172632.5, 3081187862.5, 3082111364.0, 3083942508.0, 3095748368.5, 3115054712.5, 3125271173.0, 3136767197.5, 3139454614.0, 3151460723.5, 3155680493.0, 3159900263.5, 3160490779.0, 3161081294.5, 3165700618.0, 3170319942.5, 3172758987.5, 3175198033.5, 3185674115.5, 3238683598.5, 3238702632.0, 3238721666.5, 3255278495.5, 3271835325.5, 3313783356.5, 3355731388.5, 3364551209.5, 3377852339.5, 3383910378.5, 3389968418.5, 3394760446.5, 3399552474.5, 3399619240.0, 3449926705.5, 3453256077.5, 3479035877.5, 3482775693.0, 3486515509.5, 3487137767.5, 3489681554.5, 3494860684.0, 3504567968.5, 3532386918.5, 3560205869.5, 3563122307.5, 3566038746.5, 3568889416.5, 3578160046.5, 3580701596.0, 3583243145.5, 3585596639.5, 3587950133.5, 3602873436.5, 3617796740.5, 3622473012.0, 3627149284.5, 3628634905.5, 3630120527.5, 3637333013.5, 3644545500.5, 3656716295.0, 3668887090.5, 3671888514.5, 3674889939.5, 3727343925.0, 3779797911.5, 3780700227.5, 3781602544.5, 3784670912.0, 3787739280.5, 3799510269.0, 3812506525.5, 3816451722.0, 3820396919.5, 3826147652.5, 3850168203.5, 3871479985.5, 3892791768.5, 3901328824.5, 3909865881.5, 3910519713.0, 3911173545.5, 3911298604.5, 3911423664.5, 3931049666.5, 3968181328.5, 3972437492.5, 3976693657.5, 4001712364.0, 4026731071.5, 4027903449.0, 4043792188.5, 4048596914.0, 4053401640.5, 4053763659.5, 4054125679.5, 4070230996.5, 4086336313.5, 4089852168.5, 4093368024.5, 4100750885.5, 4172474565.5, 4191459324.5, 4210444083.5, 4212639632.0, 4214835181.5, 4222997048.5, 4231158916.5, 4231185423.5, 4231211931.5, 4239190615.0, 4289499320.5]
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
