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


# Import required libraries
import argparse
import sagemaker
import boto3
import time
import pandas as pd
import json
import os
import sys
import numpy as np
import datetime
from sklearn.metrics import f1_score
from user_variable import UserDefinedVariable

# globals
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
SESSION = sagemaker.Session()
BUCKET = SESSION.default_bucket()
ROLE = UserDefinedVariable(19, os.path.basename(__file__))
SAGEMAKER = boto3.Session().client(
    service_name="sagemaker", region_name=boto3.Session().region_name
)
AUTOML_JOB_NAME = f"AUTOML-{timestamp}-i"
TRANSFORM_JOB_NAME = f"TRANSFORM-{timestamp}"
MODEL_NAME = f"MODEL-{timestamp}"
JOB_CONFIG = {
    "CompletionCriteria": {
        "MaxRuntimePerTrainingJobInSeconds": 0,
        "MaxCandidates": 0,
        "MaxAutoMLJobRuntimeInSeconds": 0,
    },
}
INPUT_DATA_CONFIG = [
    {
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{BUCKET}/data/train",
            }
        },
        "TargetAttributeName": "",
    }
]
TRANSFORM_INPUT = {
    "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": ""}},
    "ContentType": "text/csv",
    "CompressionType": "None",
    "SplitType": "Line",
}


def upload_data(verbose=False):
    start_upload = time.time()
    train_data_s3_path = SESSION.upload_data(path=args.train_file, key_prefix=f"data/train")
    test_data_s3_path = SESSION.upload_data(
        path=args.test_file, key_prefix=f"data/test"
    )
    INPUT_DATA_CONFIG[0]["DataSource"]["S3DataSource"]["S3Uri"] = train_data_s3_path
    TRANSFORM_INPUT["DataSource"]["S3DataSource"]["S3Uri"] = test_data_s3_path
    end_upload = time.time()
    upload_time = end_upload - start_upload
    if verbose:
        print(f"Upload complete in {upload_time} seconds.")
    return upload_time


def run_auto_ml(n_classes, verbose=False):
    start_training = time.time()
    SAGEMAKER.create_auto_ml_job(
        AutoMLJobName=AUTOML_JOB_NAME,
        InputDataConfig=INPUT_DATA_CONFIG,
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/output"},
        AutoMLJobConfig=JOB_CONFIG,
        RoleArn=ROLE,
        ProblemType='BinaryClassification' if n_classes == 2 else 'MulticlassClassification',
        AutoMLJobObjective={'MetricName': 'Accuracy'}
    )

    describe_response = SAGEMAKER.describe_auto_ml_job(AutoMLJobName=AUTOML_JOB_NAME)
    job_run_status = describe_response["AutoMLJobStatus"]
    while job_run_status not in ("Failed", "Completed", "Stopped"):
        describe_response = SAGEMAKER.describe_auto_ml_job(
            AutoMLJobName=AUTOML_JOB_NAME
        )
        job_run_status = describe_response["AutoMLJobStatus"]
        if verbose:
            print(describe_response["AutoMLJobSecondaryStatus"], end="\r")
        time.sleep(5)

    end_training = time.time()
    training_time = end_training - start_training
    if verbose:
        print(f"Training complete in {training_time} seconds.")
    return training_time


def retreive_model(verbose=False):
    start_model_retreival = time.time()
    best_candidate = SAGEMAKER.describe_auto_ml_job(AutoMLJobName=AUTOML_JOB_NAME)[
        "BestCandidate"
    ]
    SAGEMAKER.create_model(
        Containers=best_candidate["InferenceContainers"],
        ModelName=MODEL_NAME,
        ExecutionRoleArn=ROLE,
    )
    end_model_retreival = time.time()
    model_retreival_time = end_model_retreival - start_model_retreival
    if verbose:
        print(f"Model retreival complete in {model_retreival_time} seconds.")
    return model_retreival_time, best_candidate


def run_predictions(verbose=False):
    start_prediction = time.time()
    SAGEMAKER.create_transform_job(
        TransformJobName=TRANSFORM_JOB_NAME,
        ModelName=MODEL_NAME,
        TransformInput=TRANSFORM_INPUT,
        TransformOutput={"S3OutputPath": f"s3://{BUCKET}/inference-results"},
        TransformResources={"InstanceType": UserDefinedVariable(121, os.path.basename(__file__)), "InstanceCount": 1},
    )

    describe_response = SAGEMAKER.describe_transform_job(
        TransformJobName=TRANSFORM_JOB_NAME
    )
    job_run_status = describe_response["TransformJobStatus"]
    while job_run_status not in ("Failed", "Completed", "Stopped"):
        describe_response = SAGEMAKER.describe_transform_job(
            TransformJobName=TRANSFORM_JOB_NAME
        )
        job_run_status = describe_response["TransformJobStatus"]
        if verbose:
            print(job_run_status, end="\r")
        time.sleep(5)
    end_prediction = time.time()
    prediction_time = end_prediction - start_prediction
    if verbose:
        print(f"Prediction complete in {prediction_time} seconds.")
    return prediction_time


def download_predictions(output_path, verbose=False):
    start_download = time.time()
    predictions_path = f"inference-results/{args.test_file.split(os.sep)[-1]}.out"
    boto3.client("s3").download_file(
        Bucket=BUCKET, Key=predictions_path, Filename=output_path
    )
    end_download = time.time()
    download_time = end_download - start_download
    if verbose:
        print(f"Download complete in {download_time} seconds.")
    return download_time


def get_size(test_id):
    #size = float(os.popen(f'''aws s3 cp $(jq '.best_candidate.InferenceContainers[0].ModelDataUrl'sagemaker-runs/{test_id}.json | tr -d '"') . && gunzip -l model.tar.gz | tail -1 | tr -s ' ' |  cut -d ' ' -f3''').read().strip().split()[-1])
    size = 'N/A'
    return size


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str)
    parser.add_argument("test_file", type=str)
    parser.add_argument("test_file_with_labels", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument('max_time_per_job', type=int)
    parser.add_argument('max_candidates', type=int)
    parser.add_argument('max_time_total', type=int)
    parser.add_argument('n_classes', type=int)
    parser.add_argument('idx', type=int)
    parser.add_argument('test_id', type=str)
    parser.add_argument('-v', action='store_true')
    args = parser.parse_args()

    # NOTE: The testfile must not contain the target
    assert "targetless" in args.test_file

    if not os.path.exists('SAGEMAKER-PREDICTONS'):
        os.mkdir('SAGEMAKER-PREDICTONS')

    AUTOML_JOB_NAME += str(args.idx)

    INPUT_DATA_CONFIG[0]["TargetAttributeName"] = args.target
    JOB_CONFIG["CompletionCriteria"]["MaxRuntimePerTrainingJobInSeconds"] = args.max_time_per_job
    JOB_CONFIG["CompletionCriteria"]["MaxCandidates"] = args.max_candidates
    JOB_CONFIG["CompletionCriteria"]["MaxAutoMLJobRuntimeInSeconds"] = args.max_time_total

    path_to_inference_results = f'SAGEMAKER-PREDICTONS/{args.test_id}.csv'

    upload_time = upload_data(verbose=args.v)
    training_time = run_auto_ml(n_classes=args.n_classes, verbose=args.v)
    model_retreival_time, best_candidate = retreive_model(verbose=args.v)
    prediction_time = run_predictions(verbose=args.v)
    download_time = download_predictions(output_path=path_to_inference_results, verbose=args.v)

    labels = np.loadtxt(args.test_file_with_labels, usecols=[-1], delimiter=',', dtype='int32', skiprows=1)
    predictions = np.loadtxt(path_to_inference_results, dtype='int32')
    test_acc = np.argwhere(labels == predictions).shape[0] / labels.shape[0]
    run_dict = SAGEMAKER.list_candidates_for_auto_ml_job(AutoMLJobName=AUTOML_JOB_NAME)

    result = {
        "training_time": training_time,
        "inference_time": prediction_time,
        "test_accuracy": test_acc,
        "f1_score": f1_score(labels, predictions).tolist() if args.n_classes == 2 else f1_score(labels, predictions, average='micro').tolist(),
        "model_size": get_size(args.test_id)
    }
    json_out = json.dumps(result, indent=4, sort_keys=True, default=str)
    print(json_out)
