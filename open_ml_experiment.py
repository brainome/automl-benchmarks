import os
import json
import csv
from pathlib import Path
import sys
import numpy as np
import shutil
import argparse
import logging


def get_valid_test_id(test_id):
	if test_id == '_dis':
		return 'dis_safe'
	elif 'GAMETES' in test_id:
		return f'GAMETES-{test_id[-2:]}'
	else:
		return test_id


def get_target(test_params, data_dir):
	path_to_trainfile = f"{data_dir}{os.sep}{test_params['TRAIN_FILE']}"
	target = test_params.get('-target')
	if not target:
		with open(path_to_trainfile) as f:
			reader = csv.reader(f)
			header = next(reader)
			target = header[-1]
	return target


def get_time(test_id):
	times = [float(BTC_TIMES[test_id][model_type]) for model_type in ['RF', 'NN', 'DT', 'SVM']]
	max_time = max(int(max(times)), 1800)
	return max_time


def main(tool, suite, data_dir, clean=False):

	RESULT_DIR = f"{tool}-runs"
	PREDICTIONS_DIR = f"{tool}-predictions"
	if not os.path.exists(PREDICTIONS_DIR):
		os.mkdir(PREDICTIONS_DIR)

	if clean:
		shutil.rmtree(RESULT_DIR)

	if not os.path.exists(SPLITS_DIR):
		os.mkdir(SPLITS_DIR)

	if not os.path.exists(RESULT_DIR):
		os.mkdir(RESULT_DIR)

	if tool == 'tables':
		# clean cloud objects
		os.system('python3 helpers/run_tables.py a b c d 1 e -clean')

	with open(suite) as scenario_file:
		scenario_reader = csv.DictReader(scenario_file, delimiter='\t')

		for idx, test_params in enumerate(scenario_reader):
			test_id = test_params['TEST_ID']
			logger.info(f'Working on {test_id}. ({idx+1}/24)')
			if os.path.exists(f"{RESULT_DIR}/{get_valid_test_id(test_id)}.json"):
				continue

			trainfile = test_params['TRAIN_FILE']
			trainfile_name = trainfile.split(os.sep)[-1].replace('.csv', '')
			train_data = f"{SPLITS_DIR}/{trainfile_name}-clean-train.csv"
			test_data = f"{SPLITS_DIR}/{trainfile_name}-clean-test.csv"
			test_data_targetless = f"{SPLITS_DIR}/{trainfile_name}-clean-test-targetless.csv"
			test_data_targetless_headered = f"{SPLITS_DIR}/{trainfile_name}-clean-test-targetless-headered.csv"
			path_to_trainfile = f"{data_dir}{os.sep}{trainfile}"

			target = get_target(test_params, data_dir)
			if not os.path.exists(train_data) or not os.path.exists(test_data) or not os.path.exists(test_data_targetless) \
					or not os.path.exists(test_data_targetless_headered):
				cmd = f'python3 helpers/split_data.py {path_to_trainfile} {SPLITS_DIR} -target {target}'
				logger.info(cmd)
				os.system(cmd)
			
			n_classes = np.unique(np.loadtxt(train_data, delimiter=',', usecols=[-1], skiprows=1)).shape[0]
			n_rows = np.loadtxt(train_data, delimiter=',', usecols=[-1], skiprows=1).shape[0]
			max_time = get_time(test_id)
			
			test_id = get_valid_test_id(test_id)
			if tool == 'azure':
				wrapper_cmd = f"python3 helpers/run_azure.py {train_data} {test_data} {target} {max_time} {test_id}"
				if idx > 0:
					# after the 0-th run, we have already uploaded all of the data
					wrapper_cmd += " -du"			
			elif tool == 'sagemaker':
				max_time_per_job = max_time
				max_time_total = max_time * 2
				max_candidates = 5
				if n_rows < 500:
					print(f'Skipping {test_id} since it has only {n_rows}<500 rows.')
					continue
				wrapper_cmd = f"python3 helpers/run_sagemaker.py {train_data} {test_data_targetless} {test_data} {target} {max_time_per_job} {max_candidates} {max_time_total} {n_classes} {idx} {test_id}"
			elif tool == 'tables':
				max_time = max(1, max_time // 3600)
				if n_rows < 1000:
					print(f'Skipping {test_id} since it has only {n_rows}<1000 rows.')
					continue
				if test_id == 'nursery':
					print('Skipping nursery, since Tables raises an error due to very small minority classes.')
					continue
				wrapper_cmd = f"python3 helpers/run_tables.py {train_data} {test_data_targetless_headered} {test_data} {target} {max_time} {test_id}"
			else:
				logger.info(f'tool \"{tool}\" is not valid.')
				sys.exit(-1)

			logger.info(f'wrapper_cmd: {wrapper_cmd}')			
			result = os.popen(wrapper_cmd).read().strip()

			logger.info(result)
			with open(f"{RESULT_DIR}/{test_id}.json", 'w+') as outfile:
				print(str(result), file=outfile)
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('tool', type=str)
	parser.add_argument('suite', type=str)
	parser.add_argument('data_dir', type=str)
	parser.add_argument('-clean', action='store_true')
	args = parser.parse_args()
	assert args.tool in ['sagemaker', 'azure', 'tables']
	BTC_TIMES = json.load(open('btc-runs/btc-times.json'))
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)
	logger = logging.getLogger(__name__)
	SPLITS_DIR = 'TRAIN_TEST_SPLITS'
	if not os.path.exists(SPLITS_DIR):
		os.mkdir(SPLITS_DIR)
	main(args.tool, args.suite, args.data_dir, args.clean)