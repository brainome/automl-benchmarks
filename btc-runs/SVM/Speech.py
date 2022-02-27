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
# Invocation: brainome TRAIN_TEST_SPLITS/phpznF975-clean-train.csv -f SVM -y -split 70 -modelonly -q -o btc-runs/SVM/Speech.py -json btc-runs/SVM/Speech.json
# Total compiler execution time: 0:00:08.51. Finished on: Feb-26-2022 19:12:35.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/SVM/Speech.py
    Classifier Type:              Support Vector Machine (Linear Kernel)
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        98.33%
      Training accuracy:          97.22% (1755/1805 correct)
      Validation Accuracy:        95.74% (742/775 correct)
      Combined Model Accuracy:    96.78% (2497/2580 correct)


    Model Capacity (MEC):        401    bits
    Generalization Ratio:          0.53 bits/bit
    Percent of Data Memorized:   699.69%
    Resilience to Noise:          -0.64 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   1 |  1725    50 
                   0 |     0    30 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   1 |   737    25 
                   0 |     8     5 

    Training Accuracy by Class:
              Target |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
              ------ | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   1 |  1725     0    30    50   97.18%  100.00%  100.00%   37.50%   98.57%   97.18%
                   0 |    30    50  1725     0  100.00%   97.18%   37.50%  100.00%   54.55%   37.50%

    Validation Accuracy by Class:
              Target |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
              ------ | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   1 |   737     8     5    25   96.72%   38.46%   98.93%   16.67%   97.81%   95.71%
                   0 |     5    25   737     8   38.46%   96.72%   16.67%   98.93%   23.26%   13.16%




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
TRAINFILE = ['TRAIN_TEST_SPLITS/phpznF975-clean-train.csv']
mapping = {'1': 0, '0': 1}
ignorelabels = []
ignorecolumns = []
target = ''
target_column = 400
important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399]
ignore_idxs = []
classifier_type = 'SVM'
num_attr = 400
n_classes = 2
model_cap = 401
w = np.array([[-0.01354050040851789, -0.03721490895847903, 0.006874718149620842, 0.00041088937425595804, -0.014755123228802276, 0.01712810538763141, -0.03869346571919874, -0.054868995746487494, -0.08593452493844636, -0.028424463352158674, 0.01832037172917182, 0.03453784942595343, -0.02824349431213065, 0.04206501963237822, -0.049866108691780556, 0.06093842705353478, 0.02128181913232779, 0.01607024926792623, 0.0037769882927924307, 0.03200897614098149, 0.018060638589824624, -0.02846631687224705, -0.03726528788581209, 0.019291790063934118, 0.017362413374408237, -0.014059116575231602, -0.01668147042353916, -0.01190024020574505, 0.0014685694150459528, -0.014291694214297488, 0.040624653187159195, -0.03116569911733401, 0.0036646783231817984, -0.04177033119134525, 0.0525701335674281, -0.006168901649160362, -0.013648997430106375, 0.02051614092101817, -0.05836093619735305, 0.009539953985410637, 0.019989511161652526, 0.036533037812508246, -0.008989229262180385, 0.04215787174642662, 0.025673142495847873, 0.018369917707041106, 0.01716984370951849, -0.013631092210849597, 0.04200782760456964, -0.029890438911203337, -0.0032573020699524858, 0.02517648855630784, -0.000556197169632414, -0.011048711992661747, -0.032078965722372024, -0.03189698523365819, 0.03242296895186058, -0.055053990594405945, -0.0015336049739622392, -0.03536858860501456, -0.022686588720713458, -0.006301350139415568, 0.02773796362431318, -0.03298633009705431, -0.028741342178050074, 0.06836377091665015, -0.05613879686710471, -0.029461113896780026, 0.02041495356570903, -0.015142610878527552, 0.0195890895288479, 0.016229705788944015, 0.00021297742115021653, 0.0052046987652160795, 0.017155086650027646, -0.0005495368205716095, 0.023446572093053907, 0.0018074366078253753, 0.0363177112853036, 0.025087032385838853, 0.034342826444929565, -0.006188780509963144, -0.049328106426959326, -0.04107661844047986, -0.0064045896480675645, 0.013725603565468746, -0.023501297039251034, 0.039095208540387366, 0.0309216646603605, 0.030981568948543117, -0.04367370469051182, 0.03771560757325223, -0.03312307256705126, -0.0047762775203205555, 0.007185522407913444, -0.04018834275614381, -0.034639424048953135, -0.031062684419760324, 0.028568529561722008, -0.011189077438817512, -0.08274913841352402, 0.050432313579900245, -0.004204468662720701, -0.043802069595802365, -0.0008128862994800916, -0.03853299997410381, -0.0009510305444181002, -0.051765974633887775, 0.013188077255448903, 0.0003864086853571495, 0.016520362413657137, 0.036935871289303986, 0.022472169749252383, 0.0030932420360245184, -0.0029754378946174956, -0.033274242012047185, 0.0326848703843209, -0.01632341776020895, 0.03535140729290305, -0.011933123080024007, -0.012370228779196165, -0.006110877729030714, 0.03438186012643374, 0.07416622063609687, -0.02028345580594006, 0.023056454610968193, -0.02006218177507318, -0.014945445611983982, 0.018045716754200962, 0.01253237160028328, 0.03390109860043141, 0.05107888861265199, 0.025581053048206175, -0.006057182461408393, -0.021816772782236246, 0.007492173183693987, 0.03589614101575701, 0.0016461787994755891, 0.02613245141009706, -0.051044056197228384, -0.012400212655737316, -0.01296539361450786, -0.011627244583992774, -0.003203241662149068, -0.005089663004517105, 0.00676697465558256, 0.002866211083378315, 0.011046457574668909, 0.07446635741585561, -0.04646347291776346, -0.032626427904201824, 0.029158815130591372, -0.015160461448996843, -0.009810511609741763, -0.00444749695699167, 0.009141422953039196, 0.014925855289642841, -0.0016861573462784295, -0.019969023244990974, 0.014567265327673166, -0.0010150301514285107, -0.041028861313835235, -0.0326172664567645, -0.0009693491654594017, 0.008344700969967872, -0.030906278999493088, 0.02264317559585342, 0.0005261491647142191, 0.011955290169767234, 0.003387738007627142, -0.01031162924642924, 0.006400661772885599, -0.004734317971991022, -0.013074823866236877, 0.028229012952194935, 0.026002636572857785, 0.019746640624558578, 0.06173240815395867, -0.01428841316419585, 0.00025500007701326617, 0.03420529179156203, -0.0064865401264112945, -0.033494066579728525, -0.03770163459194776, 0.020373442843995154, -0.01097631233054895, 0.01488024994874015, -0.038958437215354046, 0.015447310945069959, -0.0061772502470407295, 0.012975901981323196, -0.021857975824849413, -0.0035527697638770214, -0.0034462626510009496, 0.010479380293485415, -0.019648166567448274, 0.06121301892108781, 0.004449442421146808, 0.015353736596226476, -0.02663871386579524, 0.013327223383956036, 0.004949380495454069, 0.008972091043259202, 0.012287949285486198, 0.08422140372725419, -0.04400438857752092, -0.009805890412994473, -0.028096288905334926, -0.008406493156786618, -0.05058470752519315, 0.01059750069002917, -0.0019716565050413733, 0.025586487162465324, 0.02589941261815737, 0.007902119574246733, -0.017664516941214396, -0.004223985120449249, -0.036839740549338945, 0.002442429709993397, -0.009075575280752167, -0.02585038301930153, 0.011076562952943323, -0.025692459010062764, 0.03439204862856675, 0.013201670880646301, 0.004457275183100388, -0.006798423888164183, -0.009556769910029708, 0.008502814149116593, -0.017219817416557447, 0.01672457605645534, 0.01790807865703835, -0.013943960581721742, -0.022045257502426974, 0.007652871710623067, 0.0191131819970023, 0.0046333047777835515, 0.002008678006595168, 0.011513885137783087, 0.014286754142702824, -0.0014986209424322965, -0.010126670883941508, -0.008928711054331451, -0.039633921637300784, 0.046094311737305396, -0.007875479601755265, -0.015531860969523033, 0.009310372082650942, 0.013910933822301835, 0.0377456123885953, 0.002896848895371686, -0.030141315397648565, 0.008253022408299188, -0.009346883523669204, -0.06830831681711175, 0.05802194880621137, -0.020251267685041287, 0.01586863504527117, 0.024126533796131576, -0.003886375621351112, 0.0012834362780568863, -0.01863894731759991, 0.061070462608548204, 0.0005173079197111218, 0.018255135073075374, -0.01687998037876265, 0.0458755951934494, 0.02153154671738954, 0.02102457069006158, 0.03129286220681003, -0.021132821042587565, 0.03783043601405907, 0.048657977657387184, 0.03631817208262082, 0.0199252141186747, -0.04400265308778696, -0.004112621708817354, 0.007147483412815127, 0.0037942238095191347, 0.0198117604367078, -0.05037396636163318, 0.0041685878225808755, 0.010761649251710548, -0.01321766543353857, -0.014124295887607465, -0.009141232511758269, -0.022849833543437118, 0.012163959415380121, 0.03097090269913649, 0.03548004785622347, 0.0009350815964959722, 0.008721983664268317, -0.006960940866474115, -0.030591115477617333, 0.020841634905963503, -0.003383184212199963, -0.02170457139520379, -0.016891724918561418, -0.015660784435023258, -0.03816451356390858, 0.012994381551224713, 0.005369672814415238, -0.011302434299263404, 0.011265046899936708, -0.009100157172466598, -0.006940462602802914, 0.032340553097380834, -0.018375236345379308, 1.9134723594143486e-05, -0.02232678009451653, 0.0022874072804892095, -0.01603809274691403, 0.01042752897212669, 0.022054856989495154, 0.01863672957015839, 0.005348766851094103, 0.06145028470640015, 0.021759741318051872, 0.015444617146795812, -0.04196648852371148, -0.04441708729304506, 0.022236230104024778, -0.030260443010027303, -0.009724284842189616, -0.02445523248642616, 0.023618610306286834, 0.010176179254511726, 0.0822510288510248, 0.018982800861538654, 0.040991165514014015, 0.011158302862942022, 0.0010559766992169148, -0.022532640429215666, 0.014459946051845045, 0.013998366387685401, 0.022264636746424645, -0.017505959405419633, 0.012894891330692024, -0.018616062801391515, 0.027038118544573247, 0.022578650886342512, -0.012627688567003026, 0.01330469285445336, 0.006721922727647325, -0.019365212316493498, -0.03886327667497759, 0.005562265583763421, -0.03714271231594698, 0.001229301899459649, -0.04136273444967036, 0.03131707517212257, -0.013005951018045837, -0.001688413046187776, -0.027248329757264506, 0.008987031265085356, -0.011312827049052052, -0.012505595820963035, -0.0224865077407507, 0.024709507032231286, 0.010835126163725235, -0.04745975532075344, -0.0019848955781387174, -0.033473317247352065, -0.005912467497339041, 0.0033485695320578115, -0.08370105890606046, -0.05277321071516784, 0.0077217442389776285, 0.02013986080328642, -0.04539724683018993, 0.015255936147991775, -8.215301674054146e-05, 0.04111451494716988, 0.011361716473110345, -0.026193233095864292, -0.0707348649503692, -0.01437795421673672, 0.03855447727992317, -0.0668183310191923, -0.0025888312719855415, -0.00032086944598539645, -0.007295394577524141, 0.03644676141648656, -0.007691588880597622, -0.03738491381904006, 0.019202039958440574, -0.029641283866843924, -0.03017747414778681, 0.0007078122627531517, -0.028650492559172135, 0.008279696912580254, 0.003956698015855456, 0.0015907335199775577, -0.07789442932479196, -0.017208105494388417, -0.008204340442900675, 0.01584217963322329, -0.009637525470402903, 0.01155090867307837, -0.03679212186787057]])
b = np.array([-0.13540068913470915])


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


def __classify(arr):
    vote = np.matmul(arr, w.reshape(-1, 1)) + b
    output = (vote > 0).astype(int)
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
