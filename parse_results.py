import re
import os
import json
import csv
import argparse
from open_ml_experiment import get_valid_test_id


MODEL_TYPES = ['RF', 'NN', 'DT', 'SVM']

def get_test_ids(test_suite):
	test_ids = []
	with open(test_suite) as f:
		reader = csv.DictReader(f, delimiter='\t')
		for row in reader:
			test_ids.append(row['TEST_ID'])
	return test_ids

def get_btc_val_acc(model_type, test_id):
	test_id = test_id.replace('-', '_')
	handle = f"btc-runs/{model_type}/{test_id}.json"
	json_dict = json.load(open(handle))
	#if args.test_suite == 'test-suites/multiclass.tsv':
	val_acc = float(json_dict['session']['system_meter']['validation_stats']['accuracy'])
	#else:
	#	val_acc = float(json_dict['session']['system_meter']['validation_accuracy'])
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


def get_btc_run_time(model_type, test_id):
	with open('btc-runs/btc-times.json') as infile:
		times = json.load(infile)
		return float(times[test_id][model_type])


def get_tool_results(test_id, tool):

	if tool == 'azure' or tool == 'tables':	
		test_id = get_valid_test_id(test_id)

	with open(f'{tool}-runs/{test_id}.json') as infile:
		results = json.load(infile)
		run_time = float(results['train_time'])
		inference_time = float(results['predict_time'])
		test_acc = float(results['test_accuracy'])
		f1 = float(results['f1_score'])
		test_acc = 100.0 * test_acc
	
	return run_time, inference_time, test_acc, f1


def get_csv_name(test_id, test_suite):
	with open(test_suite) as f:
		reader = csv.DictReader(f, delimiter='\t')
		for row in reader:
			if row['TEST_ID'] == test_id:
				return row['TRAIN_FILE'].split(os.sep)[-1].replace('.csv', '')


def get_test_acc_inference_time_and_f1(test_id, model_type, CSV_name):
	test_id = test_id.replace('-', '_')
	path_to_json = f'btc-runs/{model_type}/{test_id}.json'
	json_dict = json.load(open(path_to_json))
	test_acc = 100.0 * float(json_dict['test_acc'])
	inference_time = float(json_dict['inference_time'])
	f1 = float(json_dict['f1_score'])
	return test_acc, inference_time, f1


def build_comparison_table(tool, test_suite):
	template = "\t".join(["{}" for _ in range(13)])
	with open(f'comparison-baseline/comparison_against_{tool}.tsv', 'w+') as outfile:
		header = "test_id\t"
		header += f"btc_test_acc\t{tool}_test_acc\taccuracy_difference\t"
		header += f"btc_f1\t{tool}_f1\tf1_difference\t"
		header += f"btc_run_time\t{tool}_run_time\trun_time_difference\t"
		header += f"btc_inference_time\t{tool}_inference_time\tinference_time_difference"
		print(header, file=outfile)
		for test_id in TEST_IDS:

			json_file = f'{tool}-runs/{get_valid_test_id(test_id)}.json' if tool in ['azure', 'tables'] else f'{tool}-runs/{test_id}.json'
			if not os.path.exists(json_file):
				print(f'skipping {test_id} because {json_file} does not exist...')
				continue
			else:
				print(f'working on {test_id}...')

			CSV_name = get_csv_name(test_id, test_suite)

			btc_val_acc_dict = {model_type : get_btc_val_acc(model_type, test_id) for model_type in MODEL_TYPES}
			chosen_btc_model = get_best_model(btc_val_acc_dict)
			
			btc_test_acc, btc_inference_time, btc_f1 = get_test_acc_inference_time_and_f1(test_id, chosen_btc_model, CSV_name)
			btc_run_time = sum(get_btc_run_time(model_type, test_id) for model_type in MODEL_TYPES)

			tool_run_time, tool_inference_time, tool_test_acc, tool_f1 = get_tool_results(test_id, tool)

			accuracy_difference = btc_test_acc - tool_test_acc
			run_time_difference = btc_run_time - tool_run_time
			inference_time_difference = btc_inference_time - tool_inference_time
			f1_difference = btc_f1 - tool_f1

			print(template.format(
					'dis' if test_id == '_dis' else test_id,
					round(btc_test_acc, 2),
					round(tool_test_acc, 2),
					round(accuracy_difference, 3),
					round(btc_f1, 2),
					round(tool_f1, 2),
					round(f1_difference, 3),
					round(btc_run_time, 2),
					round(tool_run_time, 2),
					round(run_time_difference, 2),
					round(btc_inference_time, 2),
					round(tool_inference_time, 2),
					round(inference_time_difference, 2)
				), file=outfile)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('tool', type=str)
	parser.add_argument('test_suite', type=str)
	args = parser.parse_args()
	if not os.path.exists('comparison-baseline'):
		os.mkdir('comparison-baseline')
	TEST_IDS = get_test_ids(args.test_suite)
	build_comparison_table(args.tool, args.test_suite)
