import os
import shutil
import csv
import json
import importlib.util
import sys
import argparse
import time
from open_ml_experiment import get_target, get_valid_test_id

MODEL_TYPES = ['RF', 'NN', 'DT', 'SVM']
SPLITS_DIR = 'TRAIN_TEST_SPLITS'
BTC_TIMES = {}


def prepare_run_directory(model_type):
	if os.path.exists('target'):
		shutil.rmtree('target')
	predictor_dir = f"btc-runs/{model_type}"
	if os.path.exists(predictor_dir):
		shutil.rmtree(predictor_dir)
	os.makedirs(predictor_dir)


def split(test_params, data_dir):
	source_file = test_params.get('TRAIN_FILE')
	csv_name = source_file.split(os.sep)[-1].replace('.csv', '')
	train_data = f"{SPLITS_DIR}/{csv_name}-clean-train.csv"
	test_data = f"{SPLITS_DIR}/{csv_name}-clean-test-targetless.csv"
	test_data_with_labels = f"{SPLITS_DIR}/{csv_name}-clean-test.csv"
	if not all([os.path.exists(file) for file in [train_data, test_data, test_data_with_labels]]):
		target = get_target(test_params, data_dir)
		cmd = f'python3 helpers/split_data.py {data_dir}/{source_file} {SPLITS_DIR} -target {target}'
		print(f'splitting: {cmd}')
		os.system(cmd)
	else:
		print('data already split')


def make_predictor(test_params, model_type, data_dir, multiclass=False):
	test_id = test_params.get('TEST_ID').replace('-', '_')
	csv_name = test_params.get('TRAIN_FILE').split(os.sep)[-1].replace('.csv', '')
	train_file = f"TRAIN_TEST_SPLITS/{csv_name}-clean-train.csv"
	cmd = f"brainome {train_file} -f {model_type} -y -split 70 -modelonly -q -o btc-runs/{model_type}/{test_id}.py -json btc-runs/{model_type}/{test_id}.json"
	if model_type == 'DT':
		cmd += ' -rank'
	if model_type == 'RF' and multiclass:
		cmd += ' -e 5'
	start = time.time()
	print(f'making predictor: {cmd}')
	os.system(cmd)
	end = time.time()
	train_time = end - start
	raw_test_id = test_params.get('TEST_ID')
	if raw_test_id not in BTC_TIMES:
		BTC_TIMES[raw_test_id] = {}
	BTC_TIMES[raw_test_id][model_type] = train_time
	return train_time


def run_predictor(test_params, model_type):
	test_id = test_params.get('TEST_ID').replace('-', '_')
	csv_name = test_params.get('TRAIN_FILE').split(os.sep)[-1].replace('.csv', '')
	path_to_data = f'TRAIN_TEST_SPLITS/{csv_name}-clean-test.csv'
	cmd = f'python3 helpers/run_brainome_predictor.py {model_type} {test_id} {csv_name}'
	json_output = json.loads(os.popen(cmd).read().strip())
	print(f'result from running predictor: {json_output}')
	return json_output


def update_model_results(test_params, model_type, new_results):
	test_id = test_params.get('TEST_ID').replace('-', '_')
	path_to_json = f'btc-runs/{model_type}/{test_id}.json'
	old_json = json.load(open(path_to_json))
	new_json = {**old_json, **new_results}
	with open(path_to_json, 'w+') as f:
		print(json.dumps(new_json), file=f)


def main(suite, data_dir):

	assert os.path.exists(suite)
	
	for model_type in MODEL_TYPES:
		print(f'Working on {model_type}.')
		prepare_run_directory(model_type)
		with open(suite) as f:
			reader = csv.DictReader(f, delimiter='\t')
			for idx, test_params in enumerate(reader):
				if test_params.get('IGNORE'):
					continue
				split(test_params, data_dir)
				train_time = make_predictor(test_params, model_type, data_dir, multiclass=('multiclass' in suite))
				new_results = run_predictor(test_params, model_type)
				# add test information to json output
				update_model_results(test_params, model_type, new_results)
	
	with open('btc-runs/btc-times.json', 'w+') as json_out:
		print(json.dumps(BTC_TIMES), file=json_out)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('suite', type=str)
	parser.add_argument('data_dir', type=str)
	args = parser.parse_args()
	if not os.path.exists(SPLITS_DIR):
		os.mkdir(SPLITS_DIR)
	main(args.suite, args.data_dir)

