import numpy as np
import sys
import json
import os
import shutil
import time
import logging
from sklearn.metrics import f1_score

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logger = logging.getLogger(__name__)

script_path = os.sep.join(os.path.abspath(os.path.realpath(__file__)).split(os.sep)[:-1]) + os.sep

def run_predictor(model_type, test_id, CSV_name):
	path_to_predictor = f'btc-runs/{model_type}/{test_id}.py'
	path_to_data = f'TRAIN_TEST_SPLITS/{CSV_name}-clean-test.csv'
	
	new_path_to_predictor = f'{script_path}pred.py'
	if os.path.exists(new_path_to_predictor):
		os.remove(new_path_to_predictor)
	shutil.copyfile(path_to_predictor, new_path_to_predictor)

	from pred import load_data, predict, PredictorError
	start = time.time()

	logging.info(f'attempting to load {path_to_data}')
	arr, X, y, h = load_data(path_to_data, validate=True, headerless=False)
	logger.info('done loading data')
	logger.info(f'\tarr.shape: {arr.shape}')
	y = y.astype('int32')
	preds = predict(X).astype('int32')
	end = time.time()

	inference_time = end - start
	test_acc = np.argwhere(y.reshape(-1) == preds.reshape(-1)).shape[0] / y.shape[0]
	n_classes = np.unique(y).shape[0]
	result = {'test_acc' : test_acc, 'inference_time' : inference_time}
	if n_classes > 2:
		result['f1_score'] = f1_score(y, preds, average='micro')
	else:
		result['f1_score'] = f1_score(y, preds)
	print(json.dumps(result))
	if os.path.exists(new_path_to_predictor):
		os.remove(new_path_to_predictor)
	return result

if __name__ == '__main__':
	run_predictor(sys.argv[1], sys.argv[2], sys.argv[3])
