#
# BRAINOME CONFIDENTIAL
# Copyright (c) 2021-22 Brainome Incorporated. All Rights Reserved.
# NOTICE: All information contained herein is, and remains the property of
# Brainome Incorporated and its suppliers, if any. The intellectual and
# technical concepts contained herein are proprietary to Brainome Incorporated and its
# suppliers and may be covered by U.S. and Foreign Patents, patents in process,
# and are protected by trade secret or copyright law. Dissemination of this information or
# reproduction of this material is strictly forbidden unless prior written permission is
# obtained from Brainome Incorporated.

"""
bench marker for open_ml test cases on google auto ml

Submit test scenario to google auto ml,
record duration, train accuracy, test accuracy,

Seeded from
https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/main/notebooks/samples/tables/census_income_prediction/getting_started_notebook.ipynb
and
https://cloud.google.com/automl-tables/docs/samples/automl-tables-create-model

@author zachary.stone@brainome.ai
@author andrew.stevko@brainome.ai
"""
import sys
import argparse
import logging
import time
from sklearn.metrics import f1_score
import os
from pathlib import Path
import numpy as np
import contextlib
import json

from google.api_core import operation
from google.cloud import automl_v1beta1 as automl, storage

# logging defaults to stderr
# from google.cloud.automl_v1beta1 import Model
from google.cloud.exceptions import NotFound
from user_variable import UserDefinedVariable

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logger = logging.getLogger(__name__)

# globals
TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
BUCKET_NAME = UserDefinedVariable(50, os.path.basename(__file__))
COMPUTE_REGION = UserDefinedVariable(51, os.path.basename(__file__))
PROJECT_ID = UserDefinedVariable(52, os.path.basename(__file__))
AUTO_ML_CLIENT = automl.AutoMlClient()
TABLES_CLIENT = automl.TablesClient(project=PROJECT_ID, region=COMPUTE_REGION)
STORAGE_CLIENT = storage.Client()
BUCKET = STORAGE_CLIENT.get_bucket(BUCKET_NAME)
CURRENT_OPERATION: operation.Operation = None     # an ongoing operation which may need to be stopped prematurely
GCS_OUTPUT_FOLDER_NAME = f'predictions-{TIMESTAMP}'


def get_valid_test_id(test_id):
    if test_id == '_dis':
        return 'dis_safe' # no non-alphanumeric leading characters, but dis is a module
    elif 'GAMETES' in test_id:
        return f'GAMETES-{test_id[-2:]}'
    else:
        return test_id


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to a storage.bucket.blob
    https://cloud.google.com/storage/docs/
    """
    if args.v:
        logger.debug(f"uploading blob to {bucket_name} {source_file_name} {destination_blob_name}")
    # create storage.bucket.blob
    blob = BUCKET.blob(destination_blob_name)
    # upload
    blob.upload_from_filename(str(source_file_name))
    if args.v:
        logger.info(f"uploading blob to {bucket_name} {source_file_name} {destination_blob_name}")


def cloud_file_exists(file_name: str) -> bool:
    exists = storage.Blob(bucket=BUCKET, name=file_name).exists(STORAGE_CLIENT)
    if args.v:
        logger.info(f"{file_name} {'FOUND' if exists else 'NOT found'} in {BUCKET_NAME} ")
    return exists


def cloud_dataset_exists(dataset_name: str):
    """ checks if a dataset exists, returns dataset or None """
    all_datasets = TABLES_CLIENT.list_datasets()
    for dataset in all_datasets:
        if dataset_name == dataset.display_name:
            # found
            if args.v:
                logger.info("FOUND existing dataset")
            return dataset
    # name not found
    if args.v:
        logger.info("dataset NOT found")
    return None


def cloud_file_purge():
    """ delete all blobs in the bucket """
    all_blobs = BUCKET.list_blobs()
    all_blobs_list = list(all_blobs)
    if args.v:
        logger.info(f"found {len(all_blobs_list)} blobs to purge")
    BUCKET.delete_blobs(all_blobs_list)
    if args.v:
        logger.debug(f"purged BUCKET {BUCKET}")


def cloud_dataset_purge():
    """ lists all datasets and deletes them one by one """
    if args.v:
        logger.debug("purging datasets")
    all_datasets = TABLES_CLIENT.list_datasets()
    if args.v:
        logger.info(f"found {len(list(all_datasets))} datasets to purge")
    for dataset in all_datasets:
        if args.v:
            logger.debug(f"deleting dataset {dataset.display_name}")
        operation = TABLES_CLIENT.delete_dataset(dataset_name=dataset.name)
        result = operation.result()
        if args.v:
            logger.debug(f"operation result {result}")


def cloud_model_purge():
    """ lists all datasets and deletes them one by one """
    if args.v:
        logger.debug("purging models")
    all_models = TABLES_CLIENT.list_models()
    if args.v:
        logger.info(f"found {len(list(all_models))} models to purge")
    for model in all_models:
        if args.v:
            logger.debug(f"deleting model {model.display_name}")
        TABLES_CLIENT.delete_model(model_name=model.name)


def upload_data(dataset_name: str, training: Path, target: str):
    """
    Uploads csv files and creates a data set
    Parameters:
        dataset_name: name of dataset
        training:	Path to local file training data set
        target:		string column name
    return:
        create_duration_sec, dataset
    """
    global CURRENT_OPERATION
    gs_path_to_train = Path('data') / training.name
    if not cloud_file_exists(str(gs_path_to_train)):
        if args.v:
            logger.debug(f"uploading {gs_path_to_train} to cloud storage")
        upload_blob(BUCKET_NAME, str(training), str(gs_path_to_train))

    # Create dataset (not including wire time for uploads)
    start = time.time()
    # display_name = training.stem.replace('-', '_')
    dataset = cloud_dataset_exists(dataset_name=dataset_name)
    if dataset is None:
        if args.v:
            logger.debug(f'Creating dataset with name {dataset_name}')
        dataset = TABLES_CLIENT.create_dataset(dataset_display_name=dataset_name)

        # import training data into dataset
        gs_train_url = f'gs://{BUCKET_NAME}/{gs_path_to_train}'
        if args.v:
            logger.debug(f'Importing data from {gs_train_url}')
        # google.api_core.operation.Operation:
        import_data_operation = TABLES_CLIENT.import_data(
            dataset=dataset,
            gcs_input_uris=gs_train_url
        )
        # synchronous wait for operation to complete
        # as per https://github.com/googleapis/python-api-core/blob/main/google/api_core/operation.py
        if args.v:
            logger.info("importing data into dataset, please wait")
        CURRENT_OPERATION = import_data_operation
        finished = import_data_operation.result()
        if args.v:
            logger.info(f"imported data set {gs_path_to_train} {finished}")

        # update target column to be categorical, if it is numeric, so that automl does not run regression
        if args.v:
            logger.debug('Updating target column spec.')
        target_column_spec = TABLES_CLIENT.update_column_spec(
            dataset=dataset,
            column_spec_display_name=target,
            type_code=automl.TypeCode.CATEGORY,
            nullable=False,
        )
        # new column spec
        if args.v:
            logger.info(f'Updated target column spec. {target_column_spec}')

        # set target
        dataset = TABLES_CLIENT.set_target_column(
            dataset=dataset,
            column_spec_display_name=target,
        )
        # A :class:`~google.cloud.automl_v1beta1.types.ColumnSpec` instance.
        if args.v:
            logger.info(f'set_target_column. {target}')

    end = time.time()
    create_duration_sec = end - start
    if args.v:
        logger.info(f"Finished {dataset_name} in {create_duration_sec} sec")
    return create_duration_sec, dataset


def train_model(model_name: str, dataset, target: str, training_limit_hours: int):
    """ training_limit_hours # The number of hours to train the model.
    """
    global CURRENT_OPERATION
    if args.v:
        logger.debug(f"train_model {model_name} with {dataset.display_name} and target={target} limited to {training_limit_hours} hours")
    if 1 > training_limit_hours > 72:
        if args.v:
            logger.error(f"train_model hour limit {training_limit_hours} must be between 1 and 72")
        raise ValueError(f"train_model hour limit {training_limit_hours} must be between 1 and 72")
    start = time.time()

    n_classes = np.unique(np.loadtxt(args.labeled_test_data, delimiter=',', skiprows=1, usecols=[-1])).shape[0]

    try:
        model = TABLES_CLIENT.get_model(model_display_name=model_name)
        if args.v:
            logger.info("model FOUND")
    except NotFound:
        create_model_response = TABLES_CLIENT.create_model(
            model_display_name=model_name,
            dataset=dataset,
            train_budget_milli_node_hours=training_limit_hours * 1000,
            exclude_column_spec_names=target,
            optimization_objective='MAXIMIZE_AU_ROC' if n_classes == 2 else 'MINIMIZE_LOG_LOSS'
        )
        if args.v:
            logger.info(f"creating model, please wait\n {create_model_response}")
        CURRENT_OPERATION = create_model_response
        model = create_model_response.result()
        if args.v:
            logger.info("model CREATED")

    end = time.time()
    training_time = end - start
    if args.v:
        logger.info(f"Finished train_model {dataset} in {training_time} sec")
    return training_time, model


def deploy_model(model):
    """ deploy the model """
    global CURRENT_OPERATION
    start = time.time()

    if str(model.deployment_state) == 'DeploymentState.DEPLOYED':
        if args.v:
            logger.info("model has been deployed - don't overdo it.")
        pass
    else:
        if args.v:
            logger.debug(f"deploying model {model.name}")
        deploy_model_operation = TABLES_CLIENT.deploy_model(model=model)
        CURRENT_OPERATION = deploy_model_operation
        if args.v:
            logger.info(f"waiting for model {model.name} to deploy")
        deploy_model_operation.result()
        model = TABLES_CLIENT.get_model(model_name=model.name)

    end = time.time()
    deploy_time = end - start
    if args.v:
        logger.info(f"Time to deploy: {deploy_time} sec")
    return deploy_time, model


def get_predictions(model, testing: Path):
    """" get predictions from the model """
    global CURRENT_OPERATION
    if args.v:
        logger.debug("getting predictions")

    gs_path_to_test = Path('data') / testing.name
    if not cloud_file_exists(str(gs_path_to_test)):
        if args.v:
            logger.debug(f"uploading {gs_path_to_test} to cloud storage")
        upload_blob(BUCKET_NAME, str(testing), str(gs_path_to_test))


    GCS_BATCH_PREDICT_INPUT_URL = f'gs://{BUCKET_NAME}/{gs_path_to_test}'
    GCS_BATCH_PREDICT_OUTPUT_PREFIX = f'gs://{BUCKET_NAME}/{GCS_OUTPUT_FOLDER_NAME}/'
    start = time.time()
    batch_predict_operation = TABLES_CLIENT.batch_predict(
        model=model,
        gcs_input_uris=GCS_BATCH_PREDICT_INPUT_URL,
        gcs_output_uri_prefix=GCS_BATCH_PREDICT_OUTPUT_PREFIX,
    )
    CURRENT_OPERATION = batch_predict_operation
    # Wait until batch prediction is done.
    if args.v:
        logger.info("waiting for predictions to be processed")
    batch_predict_result = batch_predict_operation.result()
    results = batch_predict_result.metadata
    end = time.time()
    prediction_time = end - start
    if args.v:
        logger.info(f"Time to predict: {prediction_time}")
    return prediction_time, results


def download_predictions(test_id):
    blob_containing_preds = [x for x in list(STORAGE_CLIENT.list_blobs(BUCKET)) if GCS_OUTPUT_FOLDER_NAME in x.path and 'errors_1.csv' not in x.path][0]
    local_path_to_preds = f'tables-predictions/{test_id}.csv'
    with open(local_path_to_preds, 'wb+') as f:
        blob_containing_preds.download_to_file(f)
    return local_path_to_preds


def get_class(column_name):
    return int(column_name.split('_')[-2])


def get_sorted_idxs(test_id, n_classes):
    local_path_to_preds = f'tables-predictions/{get_valid_test_id(test_id)}.csv'
    header = np.loadtxt(local_path_to_preds, delimiter=',', max_rows=1, dtype=str)[-n_classes:]
    classes = np.array([get_class(column_name) for column_name in header])
    idxs = np.argsort(classes)
    return idxs.reshape(-1)


def get_test_results(labeled_test_data, test_id):
    local_path_to_preds = download_predictions(test_id)

    arr_preds = np.loadtxt(local_path_to_preds, delimiter=',', skiprows=1)
    arr_true = np.loadtxt(labeled_test_data, delimiter=',', skiprows=1)

    n_classes = np.unique(arr_true[:, -1]).shape[0]
    X_preds = arr_preds[:, :-n_classes]
    X_true = arr_true[:, :-1]

    concats_preds = [''.join(str(x) for x in row) for row in X_preds]
    concats_true = [''.join(str(x) for x in row) for row in X_true]

    idxs_preds = np.argsort(concats_preds)
    idxs_true = np.argsort(concats_true)

    X_preds = X_preds[idxs_preds]
    X_true = X_true[idxs_true]

    y_true = arr_true[:, -1].reshape(-1)
    y_true = y_true[idxs_true].reshape(-1)

    y_scores_unsorted = arr_preds[:, -n_classes:]
    y_scores_sorted = y_scores_unsorted[:, get_sorted_idxs(test_id, n_classes)]
    y_preds = np.argmax(y_scores_sorted, axis=1).reshape(-1)
    y_preds = y_preds[idxs_preds].reshape(-1)

    test_acc = np.argwhere(y_preds == y_true).shape[0] / y_true.shape[0]
    f1 = f1_score(y_true, y_preds) if n_classes == 2 else f1_score(y_true, y_preds, average='micro')

    return test_acc, f1


def delete_bucket():
    BUCKET.delete()


def clean_up(model, dataset):
    if model is not None:
        TABLES_CLIENT.delete_model(model_name=model.name)
    # Delete Cloud Storage objects that were created
    if dataset is not None:
        TABLES_CLIENT.delete_dataset(dataset_name=dataset.name)
    # delete_bucket()


def stop_operations():
    """ stop any current operations """
    # If training model is still running, cancel it.
    if CURRENT_OPERATION is not None:
        #CURRENT_OPERATION.refresh()
        if not CURRENT_OPERATION.done():
            if args.v:
                logger.info(f"cancelling operation {CURRENT_OPERATION.name}")
            CURRENT_OPERATION.cancel()


def main(args: argparse.Namespace):
    if args.v:
        redirector = contextlib.redirect_stderr
    else:
        redirector = contextlib.redirect_stdout
    with redirector(None):
        # ** disassemble args **
        # set log level to DEBUG (flow) else default to INFO (decision points)
        logger.setLevel('ERROR' if args.v > 0 else 'INFO')
        if args.clean:
            logger.info("cleaning cloud space rather than training")
            cloud_file_purge()
            cloud_dataset_purge()
            cloud_model_purge()
            return 0

        training_filename = Path(args.train)
        if not training_filename.exists():
            logger.error(f"training file {training_filename} not found")
            raise FileNotFoundError(training_filename)

        testing_filename = Path(args.test)
        if not testing_filename.exists():
            logger.error(f"testing file {testing_filename} not found")
            raise FileNotFoundError(testing_filename)

        target = args.target  # assume not empty
        train_limit = args.train_time
        try:
            project_name = training_filename.stem.replace('-', '_')
            if len(project_name) > 26:
                project_name = project_name[:26]
            model, dataset = None, None
            dataset_create_sec, dataset = upload_data(project_name, training_filename, target)
            training_time, model = train_model(f"model_{project_name}", dataset, target, train_limit)
            deploy_time, model = deploy_model(model)
            prediction_time, results = get_predictions(model, testing_filename)
            test_accuracy, f1 = get_test_results(args.labeled_test_data, args.test_id)
            success = True
        except Exception as e:
            logger.exception("exception detected ", exc_info=e)
            success = False
        finally:
            # clean up space
            if not args.dev and not args.clean:
                logger.debug("cleaning up training run")
                clean_up(model, dataset)
            # stop things from running away
            stop_operations()

    results = {'training_time' : training_time,
               'inference_time' : prediction_time,
               'test_accuracy' : test_accuracy,
               'f1_score' : f1,
               'model_size' : 'N/A'}
    print(json.dumps(results))
    return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=str, help="training csv filename")
    parser.add_argument('test', type=str, help="testing csv filename")
    parser.add_argument('labeled_test_data', type=str)
    parser.add_argument('target', type=str, help="target column name")
    parser.add_argument('train_time', type=int, default=1, help="maximum training time limit in hours; default 1 hour")
    parser.add_argument('test_id', type=str)
    parser.add_argument('-v', action="count", default=0, help="set verbocity to logging.DEBUG")
    parser.add_argument('-dev', action="store_true", default=False, help='dev interation - do not clean_up')
    parser.add_argument('-clean', action="store_true", default=False, help='purge test infrastructure')
    args = parser.parse_args()
    exit(main(args))
