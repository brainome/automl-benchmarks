# Brainome Daimensions(tm)
#
# The Brainome Table Compiler(tm)
# Copyright (c) 2022 Brainome Incorporated. All Rights Reserved.
# GPLv3 license, all text above must be included in any redistribution.
# See LICENSE.TXT for more information.
#
# This program may use Brainome's servers for cloud computing. Server use
# is subject to separate license agreement.
#
# Contact: itadmin@brainome.ai
# for questions and suggestions.
#
# @author: zachary.stone@brainome.ai
# @author: andy.stevko@brainome.ai

import argparse
import csv
import json
from json import JSONDecodeError
from pathlib import Path

OUTPUT_TSV = 'btc-runs/btc-results.tsv'

""" generate a TSV with results from all the BTC runs
requires 'btc-runs/btc-times.json'
requires "btc-runs/{model_type}/{test_id}.json"
"""


def get_btc_val_acc(model_type, test_id):
    try:
        model_results = get_test_model_results(model_type, test_id)
        val_acc = float(model_results['session']['system_meter']['validation_stats']['accuracy'])
    except KeyError:
        return '--'
    return val_acc


def get_best_model(model_acc_dict):
    best_model = 'RF'
    best_acc = model_acc_dict['RF']
    for model_type in MODEL_TYPES[1:]:
        model_acc = model_acc_dict[model_type]
        if model_acc > best_acc:
            best_model = model_type
            best_acc = model_acc
    return best_model


def get_from_test_results(model_type, test_id):
    try:
        model_results = get_test_model_results(model_type, test_id)
        test_acc = float(model_results['test_acc'])
        inference_time = float(model_results['inference_time'])
        f1 = float(model_results['f1_score'])
    except KeyError:
        return '--'
    return test_acc


def get_test_model_results(model_type, test_id) -> {}:
    try:
        path_to_json = Path(f'btc-runs/{model_type}/{test_id}.json')
        with path_to_json.open("r") as json_file:
            model_results = json.load(json_file)
    except (OSError, JSONDecodeError):
        return {}
    return model_results


def read_btc_times():
    try:
        path_to_json = Path('btc-runs/btc-times.json')
        with path_to_json.open("r") as json_file:
            btc_times = json.load(json_file)
    except (OSError, JSONDecodeError):
        return {}
    return btc_times


def sum_btc_times(btc_times, test_id):
    """ aggregate model build times """
    btc_run_time = 0.0      # in seconds
    times = btc_times[test_id]
    for mode_type in MODEL_TYPES:
        btc_run_time += times[mode_type]
    return btc_run_time


def build_btc_tabled_detailed():
    # load model build run times
    btc_times = read_btc_times()
    print(f"Writing {OUTPUT_TSV}")
    with open(OUTPUT_TSV, 'w+') as outfile:
        print("test_id\trun_time_sec\tchosen_model_type\tchosen_test_acc\tbest_model_type\tbest_test_acc\tRF_acc\tNN_acc\tDT_acc\tSVM_acc", file=outfile)
        for test_id in TEST_IDS:
            btc_run_time_sec = sum_btc_times(btc_times, test_id)
            test_id_key = test_id.replace('-', '_')     # NOTE clean test_id so that it can be used as dict key
            model_val_acc_dict = {model_type: get_btc_val_acc(model_type, test_id_key) for model_type in MODEL_TYPES}
            chosen_model_type = get_best_model(model_val_acc_dict)

            model_test_acc_dict = {model_type: get_from_test_results(model_type, test_id_key) for model_type in MODEL_TYPES}
            best_model_type = get_best_model(model_test_acc_dict)

            print("{}\t{:.1f}\t{}\t{:.3%}\t{}\t{:.3%}\t{:.3%}\t{:.3%}\t{:.3%}\t{:.3%}".format(
                test_id,
                btc_run_time_sec,
                chosen_model_type,
                model_test_acc_dict[chosen_model_type],
                best_model_type,
                model_test_acc_dict[best_model_type],
                model_test_acc_dict['RF'],
                model_test_acc_dict['NN'],
                model_test_acc_dict['DT'],
                model_test_acc_dict['SVM'],
            ), file=outfile)


def get_test_ids(test_suite):
    test_ids = []
    with open(test_suite) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            test_ids.append(row['TEST_ID'])
    return test_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', type=str, nargs='?', default="test-suites/open_ml_select.tsv",
                        help="tsv file in test-suites")
    args = parser.parse_args()
    #
    TEST_IDS = get_test_ids(args.suite)
    MODEL_TYPES = ['RF', 'NN', 'DT', 'SVM']
    build_btc_tabled_detailed()
