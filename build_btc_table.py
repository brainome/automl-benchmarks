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
i

import os
import json

def get_btc_val_acc(model_type, test_id):
	handle = f"btc-runs/{model_type}/{test_id}.json"
	json_dict = json.load(open(handle))
	if args.test_suite == 'test-suites/multiclass.tsv':
		val_acc = float(json_dict['session']['system_meter']['validation_stats']['accuracy'])
	else:
		val_acc = float(json_dict['session']['system_meter']['validation_accuracy'])
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
	path_to_json = f'btc-runs/{model_type}/{test_id}.json'
	json_dict = json.load(open(path_to_json))
	test_acc = 100.0 * float(json_dict['test_acc'])
	inference_time = float(json_dict['inference_time'])
	f1 = float(json_dict['f1_score'])
	return test_acc

def build_btc_tabled_detailed():
	with open('btc-runs/btc-results.tsv', 'w+') as outfile:
		print("test_id\tRF\tNN\tDT\tSVM\tbest_model\tbest_test_acc\tchosen_model\tchosen_test_acc", file=outfile)
		for test_id in TEST_IDS:
			
			model_val_acc_dict = {model_type : get_btc_val_acc(model_type, test_id) for model_type in MODEL_TYPES}
			chosen_model_type = get_best_model(model_val_acc_dict)

			model_test_acc_dict = {model_type : get_from_test_results(model_type, test_id) for model_type in MODEL_TYPES}
			best_model_type = get_best_model(model_test_acc_dict)

			print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
					test_id,
					model_test_acc_dict['RF'],
					model_test_acc_dict['NN'],
					model_test_acc_dict['DT'],
					model_test_acc_dict['SVM'],
					best_model_type,
					model_test_acc_dict[best_model_type],
					chosen_model_type,
					model_test_acc_dict[chosen_model_type]
				), file=outfile)


def get_test_ids(test_suite):
	test_ids = []
	with open(test_suite) as f:
		reader = csv.DictReader(f, delimiter='\t')
		for row in reader:
			test_ids.append(row['TEST_ID'])
	return test_ids

if __name__ == '__main__':
	TEST_IDS = get_test_ids(args.test_suite)
	MODEL_TYPES = ['RF', 'NN', 'DT', 'SVM']
	build_btc_tabled_detailed()