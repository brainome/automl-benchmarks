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

from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.core.experiment import Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.model import Model
from azureml.core.environment import Environment
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException

# --- non azure
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import argparse
import requests
import time
import os
import json
import contextlib
from user_variable import UserDefinedVariable

# globals
WORKSPACE = Workspace(workspace_name=UserDefinedVariable(28, os.path.basename(__file__)),
		   subscription_id=UserDefinedVariable(29, os.path.basename(__file__)),
		   resource_group=UserDefinedVariable(30, os.path.basename(__file__)),
		   )
CPU_CLUSTER_NAME = UserDefinedVariable(32, os.path.basename(__file__))

def create_azure_data_set(path_to_local_data):
	fname = path_to_local_data.split(os.sep)[-1]
	path_to_remote_data = f'data/{fname}'
	ds = WORKSPACE.get_default_datastore()
	return Dataset.Tabular.from_delimited_files(path=ds.path(path_to_remote_data))


def create_compute_cluster(args):
	start = time.time()
	# Choose a name for your CPU cluster	
	vm_size = UserDefinedVariable(44, os.path.basename(__file__))
	max_nodes = UserDefinedVariable(45, os.path.basename(__file__))

	# Verify that cluster does not exist already
	try:
		compute_target = ComputeTarget(workspace=WORKSPACE, name=CPU_CLUSTER_NAME)
		if args.v:
			print("Found existing cluster, use it.")
	except ComputeTargetException:
		compute_config = AmlCompute.provisioning_configuration(
			vm_size=vm_size, max_nodes=max_nodes
		)
		compute_target = ComputeTarget.create(WORKSPACE, CPU_CLUSTER_NAME, compute_config)
		if args.v:
			print('Created compute cluster.')
	compute_target.wait_for_completion(show_output=args.v)
	end = time.time()
	return end - start, compute_target


def load_data(args):
	start = time.time()
	destination = 'data'
	ds = WORKSPACE.get_default_datastore()
	ds.upload(
		src_dir=os.sep.join(args.train.split(os.sep)[:-1]), target_path=destination, overwrite=True, show_progress=False
	)
	end = time.time()
	if args.v:
		print('Loaded data.')
	return end - start


def run_auto_ml(args, compute_target):
	if args.v:
		print('starting automl')
	start = time.time()
	automl_settings = {
		"enable_early_stopping": True,
		"max_concurrent_iterations": 4,
		"max_cores_per_iteration": -1,
		"featurization": "auto",
	}
	# task can be one of classification, regression, forecasting
	experiment_name = args.test_id
	train_data = create_azure_data_set(args.train)
	automl_classifier = AutoMLConfig(task='classification',
									 primary_metric='accuracy',
									 compute_target=compute_target,
									 experiment_timeout_minutes=args.max_time,
									 training_data=train_data,
									 label_column_name=args.target,
									 validation_size=0.3,
									 **automl_settings)
	experiment = Experiment(WORKSPACE, experiment_name)
	remote_run = experiment.submit(automl_classifier, show_output=args.v)
	remote_run.wait_for_completion()
	end = time.time()
	if args.v:
		print('Done running automl.')
	return end - start, remote_run


def deploy_model(args, remote_run):
	if args.v:
		print('deploying model')
	best_run = remote_run.get_best_child()
	model_name = best_run.properties["model_name"]
	model = remote_run.register_model(
		model_name=model_name, description=str(args.test_id), tags=None
	)
	script_file_name = f"inference/{args.test_id}.py"
	script_file_name = script_file_name.replace('-', '_')
	inference_config = InferenceConfig(entry_script=script_file_name)
	aciconfig = AciWebservice.deploy_configuration(
		cpu_cores=2,
		tags={"area": "bmData", "type": "automl_classification"},
		description=str(args.test_id),
	)
	aci_service_name = model_name.lower()
	try:
		aci_service = Model.deploy(WORKSPACE, aci_service_name, [model], inference_config, aciconfig)
		aci_service.wait_for_deployment(show_output=args.v)
	except WebserviceException:
		aci_service = AciWebservice(WORKSPACE, aci_service_name)
		if args.v:
			print('Webservice exists.')
	if args.v:
		print('done deploying model')
	return aci_service


def delete_aci_service(aci_service):
	aci_service.delete()
	while True:
		try:
			aci_service.check_for_existing_webservice(WORKSPACE, aci_service.cname)
			break
		except WebserviceException:
			time.sleep(1)
	if args.v:
		print('webservice deleted')

def predict(args, aci_service):

	start_time = time.time()

	test_dataset = create_azure_data_set(args.test)
	X_test = test_dataset.drop_columns(columns=[args.target])
	y_test = test_dataset.keep_columns(columns=[args.target], validate=True)
	
	X_test = X_test.to_pandas_dataframe()
	y_test = y_test.to_pandas_dataframe()


	if args.v:
		print('before resp')

	if args.test_id in ['MiniBooNE', 'SEA-50000']:

		batch_size = 10000
		n_rows = X_test.shape[0]
		start = 0
		y_pred = []

		while start < n_rows:

			end = min(start + batch_size, n_rows)
			X_test_batch = X_test[start:end]

			X_test_json = X_test_batch.to_json(orient="records")
			data = '{"data": ' + X_test_json + "}"
			headers = {"Content-Type": "application/json"}
			resp = requests.post(aci_service.scoring_uri, data, headers=headers)
			y_pred_batch = json.loads(json.loads(resp.text))["result"]
			y_pred.extend(y_pred_batch)

			start = end

	else:
		X_test_json = X_test.to_json(orient="records")
		data = '{"data": ' + X_test_json + "}"
		headers = {"Content-Type": "application/json"}
		resp = requests.post(aci_service.scoring_uri, data, headers=headers)
		y_pred = json.loads(json.loads(resp.text))["result"]

	y_test = np.array(y_test.values).reshape(-1)
	y_pred = np.array(y_pred).reshape(-1)
	if args.v:
		print(y_test[:10])
		print(y_pred[:10])
		print(np.unique(y_test), np.unique(y_pred))
	test_accuracy = np.argwhere(y_pred == y_test).shape[0] / y_pred.shape[0]
	n_classes = np.unique(y_test).shape[0]
	if n_classes > 2:
		f1 = f1_score(y_test, y_pred, average='micro')
	else:
		f1 = f1_score(y_test, y_pred)
	if args.v:
		print('Done with predictions.')
	
	delete_aci_service(aci_service)
	end_time = time.time()
	
	return end_time - start_time, test_accuracy, f1


def get_size(args, remote_run):
	if not os.path.exists('inference'):
		os.mkdir('inference')
	best_run = remote_run.get_best_child()

	model_name = best_run.properties["model_name"]
	if args.v:
		print('best model name: ', model_name)
		print(best_run.properties)

	script_file_name = f"inference/{args.test_id}.py"
	script_file_name = script_file_name.replace('-', '_')
	best_run.download_file("outputs/scoring_file_v_1_0_0.py", script_file_name)
	model_size = os.path.getsize(script_file_name)
	if args.v:
		print(f'Size: {model_size}.')
	return model_size


def main(args):
	if args.v:
		redirector = contextlib.redirect_stderr
	else:
		redirector = contextlib.redirect_stdout
	with redirector(None):
		setup_time, compute_target = create_compute_cluster(args)
		if not args.du: # du = don't upload
			load_time = load_data(args)
		else:
			load_time = 'N/A'
		train_time, remote_run = run_auto_ml(args, compute_target)
		model_size = get_size(args, remote_run)
		aci_service = deploy_model(args, remote_run)
		predict_time, test_accuracy, f1 = predict(args, aci_service)
		results = {'training_time' : train_time,
				   'inference_time' : predict_time,
				   'test_accuracy' : test_accuracy,
				   'f1_score' : f1,
				   'model_size' : model_size}
	json_out = json.dumps(results)
	print(json_out)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('train', type=str)
	parser.add_argument('test', type=str)
	parser.add_argument('target', type=str)
	parser.add_argument('max_time', type=int)
	parser.add_argument('test_id', type=str)
	parser.add_argument('-v', action='store_true')
	parser.add_argument('-du', action='store_true')
	parser.add_argument('-po', action='store_true')
	args = parser.parse_args()
	main(args)