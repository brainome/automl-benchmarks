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
# Invocation: brainome TRAIN_TEST_SPLITS/dis-clean-train.csv -f NN -y -split 70 -modelonly -q -o btc-runs/NN/_dis.py -json btc-runs/NN/_dis.json
# Total compiler execution time: 0:00:31.16. Finished on: Feb-26-2022 18:40:34.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        btc-runs/NN/_dis.py
    Classifier Type:              Neural Network
    System Type:                  Binary classifier
    Training / Validation Split:  70% : 30%
    Accuracy:
      Best-guess accuracy:        98.45%
      Training accuracy:          98.53% (1820/1847 correct)
      Validation Accuracy:        98.48% (781/793 correct)
      Combined Model Accuracy:    98.52% (2601/2640 correct)


    Model Capacity (MEC):        109    bits
    Generalization Ratio:          1.89 bits/bit
    Percent of Data Memorized:   199.79%
    Resilience to Noise:          -1.21 dB







    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |  1819     0 
                   1 |    27     1 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |   780     0 
                   1 |    12     1 

    Training Accuracy by Class:
               class |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |  1819    27     1     0  100.00%    3.57%   98.54%  100.00%   99.26%   98.54%
                   1 |     1     0  1819    27    3.57%  100.00%  100.00%   98.54%    6.90%    3.57%

    Validation Accuracy by Class:
               class |    TP    FP    TN    FN     TPR      TNR      PPV      NPV       F1       TS 
               ----- | ----- ----- ----- ----- -------- -------- -------- -------- -------- --------
                   0 |   780    12     1     0  100.00%    7.69%   98.48%  100.00%   99.24%   98.48%
                   1 |     1     0   780    12    7.69%  100.00%  100.00%   98.48%   14.29%    7.69%




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
TRAINFILE = ['TRAIN_TEST_SPLITS/dis-clean-train.csv']
mapping = {'0': 0, '1': 1}
ignorelabels = []
ignorecolumns = []
target = ''
target_column = 29
important_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
ignore_idxs = []
classifier_type = 'NN'
num_attr = 29
n_classes = 2
model_cap = 109
w_h = np.array([[-0.17019329965114594, -0.17371150851249695, -0.09757493436336517, 0.13302259147167206, 0.10776734352111816, 0.08884945511817932, -0.11236917227506638, 0.23828041553497314, 1.4873377084732056, -0.7160223126411438, 0.3641079366207123, -0.6737018823623657, -0.79111248254776, 0.16824716329574585, 0.16280953586101532, 0.6335561871528625, 0.43055275082588196, -0.18321870267391205, -0.44868743419647217, 0.1628582924604416, 0.9908884167671204, -0.13596205413341522, 0.5119574069976807, 0.2259284406900406, 0.1443822979927063], [-0.26354271173477173, 0.07581614702939987, 0.6701077818870544, 0.06644116342067719, 0.03098355606198311, -0.05020115524530411, -0.6802775859832764, -0.615182638168335, 0.2617432177066803, 0.4563547968864441, 0.7039390206336975, 0.16057372093200684, 0.33071327209472656, 0.12266997992992401, 1.338613748550415, 0.3043883740901947, 0.9432200789451599, -0.37885600328445435, -0.29884135723114014, -0.21273642778396606, 0.020006734877824783, 0.9718884825706482, -0.11564117670059204, 0.21716369688510895, 1.1034988164901733], [-0.10344455391168594, -0.1635366827249527, -0.2583828568458557, 0.09998898208141327, 0.13712148368358612, -0.0015201130881905556, 0.31928780674934387, 0.07300689816474915, -0.8946552872657776, -0.37628525495529175, -0.6969021558761597, 0.520952045917511, 0.2794416546821594, 0.31941232085227966, -0.6168922781944275, -0.6780195236206055, -0.4720117151737213, 0.5809774398803711, 0.09717618674039841, 0.1280457079410553, -0.5803282260894775, 0.09187175333499908, -0.38500529527664185, -0.6178987622261047, -0.20875345170497894], [0.2097092568874359, 0.0029491009190678596, -0.529596745967865, -0.06505429744720459, -0.037187360227108, 0.01268984004855156, -0.40490639209747314, 1.0598064661026, 1.0885009765625, -0.3887064456939697, 0.6142860651016235, -0.09365242719650269, -0.2617805302143097, 0.5668905973434448, -0.1542484611272812, 0.2214704155921936, -0.04288496822118759, 0.05491911619901657, -0.1328577697277069, 0.1963171511888504, 0.9298771023750305, -0.37280088663101196, -0.3464771807193756, 0.4340231418609619, 0.181315079331398]])
b_h = np.array([-1.1009615659713745, -0.21217407286167145, 0.18040494620800018, 0.3926336467266083])
w_o = np.array([[-0.542689859867096, -0.05275065824389458, 0.5176296234130859, -0.237497478723526]])
b_o = np.array(-0.7641032338142395)


class PredictorError(Exception):

    def __init__(self, msg, code):
        self.msg = msg
        self.code = code

    def __str__(self):
        return self.msg
def __transform(X):
    mean = np.array([46.683811586356256, 1.2490525175961018, 0.1321061180292366, 0.014618299945858148, 0.014076881429344884, 0.03519220357336221, 0.016242555495397944, 0.012994044396318355, 0.014076881429344884, 0.05847319978343259, 0.06442880346507851, 0.003789929615592853, 0.01028695181375203, 0.028153762858689767, 0.0005414185165132648, 0.04981050351922036, 0.9014618299945858, 126.95397942609637, 0.7969680563075258, 31.066053059014617, 0.932322685435842, 120.80184082295615, 0.8841364374661613, 66.9837574445046, 0.8852192744991879, 110.47103410936654, 0.0, 0.0, 3.2972387655657824])
    components = np.array([[-0.011182236184822841, -5.855627141987907e-05, -0.0003526537286129242, 4.655330304863249e-06, -5.182472912002636e-06, -3.465029511612208e-05, -3.024311675103075e-05, -1.522449605827098e-05, 1.982207189726215e-06, 1.945872071061679e-05, -7.156501025157715e-06, 2.6463418038442865e-06, 1.4785412843505084e-05, -1.6686022309775706e-05, 1.9412290292601877e-06, -0.00012773194235212316, -0.0010466602058918, 0.36418712602465186, -0.0007092579563201726, 0.027697961147700182, -0.0009473508501913479, 0.6570865955899468, -0.00120593119442586, 0.09502546825881777, -0.001201852482204772, 0.6524387965657436, -0.0, -0.0, 0.0009790865571344576], [-0.017787004759477462, -0.0006824484391566273, -1.7784664802603126e-05, 4.700205863342314e-05, -9.232196211878663e-05, -4.3689779363203505e-05, -7.135195955149787e-05, -2.8146475482292087e-05, -4.799565963439522e-05, 8.552318852354121e-05, -5.9688483438341817e-05, 1.9557354232916928e-05, 4.1743185120522905e-05, -2.7141343693026973e-05, -1.1290723332761816e-05, 3.3087699801775685e-06, -0.0017790772977169288, 0.8547823724628584, -0.0012447231081362709, 0.058882647798465415, -0.0008420544350938219, -0.4944314864197405, -0.0010428946176357903, 0.14519333565870893, -0.0010552079926355994, -0.003143305140165151, 0.0, 0.0, 0.001395266547334909], [0.01221050485658349, 0.0007403904469233752, -0.0002717117416341066, 6.988181122610651e-06, -5.248929538798376e-05, 0.00014359273003907398, -0.00020163804602826957, -5.924674749220502e-05, -6.502284196446655e-05, -0.00012428091548612684, -8.369393252479072e-05, -1.2225171051292409e-05, 9.950730607188007e-06, -7.440389945434977e-05, -4.380986552021176e-06, 0.00016829966390807523, -0.0007733701089645426, 0.3465753562925919, -0.00043651102233955105, 0.002899549789720499, -0.00016105232833418947, 0.5611709583900327, 0.000767990305806244, -0.1450232958810708, 0.0007725735074590405, -0.7374145615395945, 0.0, 0.0, -0.0006561026179437266], [-0.21948422659265948, -0.002214810552511894, 0.0007319544755730633, -0.00022930885528404837, 0.0003112830385638858, -0.0004015591547659537, 0.0008737359953949166, -4.572970609929858e-05, -0.00011704392828300955, -0.0002884414593540448, 0.0006913582510758097, -5.8287925915035934e-05, 5.759612523466648e-05, 0.0003761672933453139, 4.734598068416466e-06, -1.80853853047068e-05, -0.0025372138493551934, -0.12556178374332744, -0.003988726006346428, 0.2869054990775731, -0.00418074374361054, 0.09147299564243956, -0.006748385286216428, 0.9035870448148283, -0.0066477642899148304, -0.16962299971266956, 0.0, 0.0, 0.0007269929256194155], [0.9114033476368745, -0.000767333990469634, -0.00043776427862729706, -0.00014399086625339362, -0.00018989410200391768, 0.0009877988597541136, -0.0004194512104563928, -0.00020053874672959127, 0.00015158243980430058, 0.0005487255781412282, -5.446187792658995e-05, -0.0001841350999779703, -0.00037234314877806584, -3.0893168840214735e-05, -3.496475975120031e-05, -0.0014139720303860176, 0.0006413568439492735, -0.0028227181978974355, 0.00396311909449983, -0.2777838631559234, -0.001245128807430115, 0.017916540235322076, -0.0033589312088641576, 0.3011552014007524, -0.003314583602977346, -0.0329345725726407, -0.0, -0.0, 0.005242579194962109], [0.3471074066639187, -0.0005503544896056495, 0.003214156930876207, 0.00011374365605673857, -0.00017988330555277377, -0.00023496841724363637, -0.00045553287666458785, 3.188569731652063e-05, 0.00018841669303830782, 0.0010830350236733252, -0.00030258972862976175, -4.775041069821033e-05, 0.00020758096610113238, 0.00019368102204300148, -3.829109332725166e-05, -0.0013833356225638224, -0.0032483121311945385, -0.02866830557283095, -0.017557650526878178, 0.9142107440027989, -0.0018571827116547427, -0.013104852184577735, 0.0013322335590478479, -0.20383423426202368, 0.0012276289067542463, 0.025985638793728085, 0.0, 0.0, 0.013345861640728922], [0.009426252433706843, 0.032836978302666915, -0.05306765189183293, -0.0031197415867231975, -0.004693306016489335, 0.0002042530292056984, 0.01856141470879509, -0.005060449514408917, -0.008371671133241498, -0.0015718284386848096, -0.01764625606118485, 0.004700207062257145, -0.0030360872394779477, -0.003634620622551145, -0.00011503895899959865, 0.07653893079930627, 0.02325273425682138, 0.0008695610809309808, 0.02062493567786525, 0.011753275186353395, 0.005139778782224464, -0.00041773245450326877, 0.025811379524719708, 0.00042207461694695466, 0.02529822827429198, 0.0011785524232681547, -0.0, -0.0, -0.9934227297136814], [-9.067970900097064e-07, -0.9944095023199442, 0.036776723723584606, -0.01083783704705069, -0.00022280342116230017, -0.0039192058782417365, 0.014183117954120193, 0.008169385676179966, 0.001109753284200609, 0.020375940008785662, 0.023092850883597183, 0.0002826394960534493, -0.0018721162179142882, 0.020669112239681046, -0.0014786018512353166, -0.029133725857827673, 0.028494723235256987, 8.933118297388564e-05, 0.018758880443439194, -0.0002937495151088152, 0.0030667092387685322, 0.0005725583587672956, 0.05000808283620701, -0.0015530958770381316, 0.04847379972585307, -0.00015906986895558177, -0.0, -0.0, -0.0337397688433729], [-0.00014008547084997717, 0.04592666991810893, 0.9787328901078085, 0.010616748536702724, -0.006741994474904329, -0.028458174880356443, 0.018388834311752617, -0.00762794491346224, 0.021918551393019844, 0.1572836240584802, -0.025724506805613247, -0.001962095831706447, -0.00789335308754357, -0.03159044098751175, -0.00015233679999828473, -0.0038241376625877182, 0.061892399601989355, 0.0005852473555331978, -0.021467772286553825, -0.0028159891837209144, 0.03600023583504712, 0.0004249458767215469, 0.04389682762386451, 0.0012241434438263238, 0.04363678655844562, 7.042734193178358e-05, 0.0, 0.0, -0.04712325253439088], [0.0010684229623459153, 0.06700926017439045, -0.12674195529570673, -0.018282346287093736, 0.030815140501025597, 0.003839701254368832, 0.10750427972448019, 0.00595120409383289, 0.02724346907630153, 0.23999454573454018, 0.05129381960709953, 0.0010957661771449545, -0.0051538405466284385, -0.027429141700870904, 4.396560606090489e-05, -0.05344283352724582, 0.4775423274343288, 0.0013437523529695758, 0.15708057791604488, 0.00568958224670776, 0.45753733663334967, 0.0010006154053472458, 0.46565874322142703, 0.009747541210983958, 0.46986437045440316, -0.00021733161813868983, 0.0, 0.0, 0.046312144877253186], [0.0007308330656792251, 0.03078405455377037, 0.030663604925107297, 0.0005328158824885095, 0.08655705942906215, -0.0711886228144017, 0.09097068915977849, -0.0033491049840999485, 0.04244842313689732, 0.015433793047620208, 0.9441258079249433, 0.004456288847247497, -0.005086813018083359, 0.06113259141550708, -0.0004529321244230695, -0.08585185774247124, -0.2241031637789049, -0.00011514817688682934, 0.10111786700327852, 0.000887480832820679, -0.06677438694477607, -0.00010973380459920943, 0.061387962604429476, -0.0005001436691026773, 0.058836433331926015, 0.00014829807812727827, -0.0, -0.0, -0.02366412789777575], [-0.001429411596053909, -0.008265327538504794, -0.13180281407740163, -0.03807664472471874, -0.01091974323646332, 0.016307921158096268, -0.08302346290404895, -0.01729996989779794, 0.01624726140044944, 0.9330022130210602, 0.03928753919037146, -0.0015439051627732289, -0.018153113573487343, -0.06151401448198822, -0.0005646605210053497, 0.05031021364978101, 0.024580926218882995, -0.00027979214463010244, -0.10766010434362738, -0.0023319021273880552, -0.12156753102233746, -0.00021394209413880039, -0.18324630376343495, -0.0028783294147872807, -0.1843796702231899, -0.0001543314062049879, 0.0, 0.0, -0.004470575653535316], [0.0012416718257929537, -0.026484776094234993, 0.009353468026241322, -0.05931959426393529, 0.00262295463435935, -0.263731558594791, -0.07691059884919753, -0.006094109184508297, 0.025741157194528148, -0.0730001615606851, 0.12270803743097924, 0.014549372135483634, -0.024279145698669003, 0.0034539421728764686, -0.002855061111111554, 0.921059311568496, 0.1798839763670057, 0.00025774452893124913, 0.045240882487248626, 0.0016351384499845851, 0.06362318481669875, 0.00010865542097105524, -0.05752822217916491, -0.0003327766992689837, -0.05896173021752649, 2.8943854073673086e-05, 0.0, 0.0, 0.06866579637987125], [0.0011111351572067512, 0.014367863635856582, -0.018743466425464785, 0.11086678157359246, -0.02179096380538571, -0.2517089935488339, 0.14223532624585744, -0.02787585415313773, -0.017868427873663943, 0.19162128921015842, -0.2588426373725034, -0.01132328275510856, 0.061915376461414214, 0.2506171311918943, 0.003674611463100993, 0.16780185436466868, -0.6693149425726852, -0.0010121725736174183, 0.1670432973905383, -0.0003878321132451492, -0.10151787681676974, 8.292121808480574e-05, 0.32777225211528926, 0.003506454591608977, 0.330294896112031, 0.00015654471090703723, -0.0, -0.0, 0.024402827699398586], [-0.00031515409225952234, -0.004461567452656846, 0.02939622385244545, 0.07577680690973917, -0.007177059705232341, 0.9223420149430591, 0.03682008121202148, 0.003050402825341085, -0.014768944100253848, 0.017137726958776568, 0.04369155557758389, -5.29683688433856e-05, 0.0008010452005783484, 0.053770250386906755, 0.0010924941122795774, 0.30814679728344824, -0.13781078123591572, -0.0002771025631826652, -0.004850786303592754, -0.0003172222249392466, -0.03877714259064759, -1.0326314483614108e-05, 0.10002343301233751, 0.0012072792434974632, 0.09950404035708786, 0.0001750502328335115, 0.0, 0.0, 0.023402765266517897], [-0.00022337819465755192, 0.019272269051181727, 0.023289158378997605, -0.04069168696799485, -0.04569712930266732, 0.02616643895276047, -0.02776501107584745, 0.009942280611215557, -0.024436647591979832, 0.024446356777388545, 0.0049813120279395395, -0.0002905766590016311, -0.014786505604329536, 0.9548605292161966, -0.0023202250459400937, -0.06021793823366735, 0.21687147688632236, 0.000480087768326937, 0.04362528429027844, 0.0015136528958871747, 0.04261674066078295, 9.135622159270803e-06, -0.1126787061160108, -0.0014918613655876857, -0.11590120672325502, -5.2831788635074884e-05, 0.0, 0.0, -0.008498811372858182], [0.00044508847311540493, -0.02986964497035054, 0.010174338120535316, 0.44044484614938345, -0.07021788297791755, 0.019963867984531387, -0.1569684706503345, -0.022708980552712317, -0.04775177027294551, 0.037834804295637986, 0.019767460219773155, -0.0029609924070079095, 0.06262255858321557, -0.03410670742420775, 0.012077007542231592, -0.017167484255957374, -0.2579484007092066, -0.00013804191348288663, 0.24477614737852382, 0.0054459859360335705, 0.7226624494901436, 0.00019616543210773836, -0.2305709416826909, -0.0013213784123619627, -0.2504737588062994, -0.0001271619253487774, -0.0, -0.0, -0.016326995777615372], [-0.0006137836638456004, -0.003471132446503142, -0.02791507037046246, 0.3759459507454003, -0.3463274387502303, -0.08042667058748228, 0.04446571263566503, 0.31905885070339796, 0.0060965806509071675, -0.01912392273165778, 0.0966523745370558, -0.0037186355742674206, 0.06895251059624193, 0.060900743484258876, 0.019755149271898833, 0.04154424664290268, -0.0037509708870436557, -0.00030596519353900763, -0.7610881014600422, -0.013734458949050718, 0.0807797071277666, 7.074790553646963e-05, 0.09755997808496272, 0.002746148938115596, 0.10849104059556292, -5.723144241436426e-05, 0.0, 0.0, -0.008310627863419021], [0.00029429653500871354, -0.005859287570674366, -0.0004287411537424052, 0.23416215835546378, 0.6133713410974352, -0.03157921325334233, 0.47375280191569974, 0.023301224876617576, -0.5483051694006847, 0.023204159282106323, -0.032816948658091816, -0.007688970983108017, 0.021484169907000566, 0.014564995260139685, 0.007456698785462388, 0.04800912450390916, 0.07386204940123929, 0.00016330057677839884, -0.14504910777259386, -0.0023645479742250082, 0.010974424264512578, -6.653872632065614e-05, -0.0709346770436032, -0.0013127291384161302, -0.07649690560551363, -2.5126600051826336e-06, -0.0, -0.0, 0.008589707561615822], [-0.00014534684652791163, -0.009409190293267344, -0.024714388169462417, 0.23661027402799048, 0.2919504636849128, -0.0013964489435074816, 0.3479446996270947, -0.3032829419255853, 0.780970055348626, -0.013507130047880903, -0.06331404216330458, -0.007468978382580378, 0.029433995988372565, 0.04094154998891233, 0.00880863107907534, 0.016816527093035445, 0.038922780402292995, 9.913207913891058e-05, -0.13201419300167425, -0.0021577782305452872, 0.018550982715837272, -5.49587928191628e-05, -0.06357042403912175, -0.0009737743453089731, -0.06803281240121305, -6.563950216725338e-05, 0.0, 0.0, -0.0026284343401096837], [0.00018420530139483124, 0.010036870051092592, -0.0005047559524718285, 0.5922469805255826, 0.1416584336976064, -0.0231428889926518, -0.24691028986177477, 0.45186254129387193, 0.1256391518282968, 0.012044189488749297, -0.019283547455428117, 0.005821924388642349, -0.15410729438220397, -0.015201294296617021, 0.022851531619056138, -0.01624385267382073, 0.20679743550377622, 0.00029930260174237716, 0.3587269064109947, 0.006317368235814075, -0.3830063291915666, -3.579646862613123e-05, 0.04157510774742406, -0.00042960182122118567, 0.04874104221702955, -1.0257282050361658e-05, 0.0, 0.0, 0.0021789575206596053], [0.0002769499940806587, 0.0008547855788015642, 0.009165319293222752, -0.4189942151843413, 0.28885575932334845, 0.030821848560611837, 0.08707947237750555, 0.7493712852818849, 0.2562688142372806, 0.016594567519351344, -0.05793365398598648, -0.011805867371473164, -0.009344503815266471, 0.012271686037077045, -0.020447538655836508, 0.007099140274922159, -0.18100123828045833, -9.039661280695807e-05, -0.01855057387959527, -0.000357049014485556, 0.24409426351854252, 2.7766590529626075e-05, -0.057146724156415585, -0.0003881202022743914, -0.06802502979686159, -3.9588857913374084e-05, 0.0, 0.0, -0.009951646726722538], [0.0003489508849731223, 0.0036892051541908646, -0.005035159402414172, 0.02437184028092305, -0.5461946555344567, 0.00642163474755113, 0.7074824990472588, 0.18174970479657301, -0.01640660781980554, 0.02350827199679396, -0.007451918455644771, -0.0037133881030760587, -0.016577728611642063, -0.07462425814935876, -0.004849328043415145, 0.018958649599473465, 0.10252160111887974, 0.0002942643366301505, 0.322154309732551, 0.006008990296043683, -0.07800476913564974, -5.7358865750222125e-05, -0.13445410505108435, -0.002952978359226058, -0.14650168048027912, -3.863847275532367e-05, 0.0, 0.0, 0.018723629434634206], [0.00030454660963560447, 0.0018759290694329122, 0.008562577399390873, 0.012902442498175619, 0.023506210221065152, 0.010596859371683174, -0.051611175851774216, 0.07022998591510857, 0.015324576329537817, 0.007624393063350168, 0.015986556215049973, -0.00031635934544759013, 0.9801899597816589, -0.009775978372820899, -0.0017965121231486305, 0.005890824061893341, 0.09906664981014895, 0.00012866911571940363, 0.09723479480463904, 0.0016001338389967964, -0.10309945812771648, -9.586400471772532e-06, -0.008875811630009542, -0.0003234369966514069, -0.007718111679559626, -8.06701737228263e-06, 0.0, 0.0, -0.0014005625317599078], [0.00011920216627873266, 0.00029430553867510606, 0.0014530535642981526, 9.606972968138035e-05, 0.005243620968519728, 0.001360318326140789, 0.013048142898587748, 0.005713326115857771, 0.0030418390953212424, 0.004995824505449363, -0.009792970157929702, 0.9996418372658686, 0.0029120063832174447, 0.0031781824811590196, -0.0005848985043869411, -0.010496839465310409, -0.012480787478841855, -4.332550360382179e-05, -0.00496910964178953, -0.00018145239765191985, 0.005056876961997751, 2.4084770781484037e-05, -0.0003089711878059595, 0.00010123809270133982, 0.0013257420651282175, -3.1706459471698276e-05, 0.0, 0.0, 0.003818393408726486]])
    explained_variance = np.array([15592.668863049576, 5735.259144809599, 5324.196792579755, 885.0759133866519, 417.34544284673274, 342.7017030053049, 1.0906240601397597, 0.2590965954279679, 0.10719683169061121, 0.06979288272381855, 0.0625766118062138, 0.05286810547704638, 0.0398751253420017, 0.03393007014293396, 0.03263162496473658, 0.026417498098699276, 0.015904544855129064, 0.01523253278953592, 0.014342991566786627, 0.013264276729660102, 0.01274980618197735, 0.012243546056181632, 0.011234752855700721, 0.009824753988990926, 0.0037162282658409966])
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
