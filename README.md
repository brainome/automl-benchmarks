<!--
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
# @author: andy.stevko@brainome.ai
# @author: zachary.stone@brainome.ai
-->

# brainome/automl-benchmarks
AutoML training & batch prediction benchmarks comparing Brainome, Google Vertex AI, AWS Sagemaker, and Azure AutoML engines

## Published results white paper 
https://www.brainome.ai/automl-compare/

## Table of contents

- [Published results white paper](#published-results-white-paper)
- [Methodology](#methodology)
- [Setup](#setup)
  - [4 Separate Virtual Environments](#4-separate-virtual-environments)
  - [Fetch seed data from Open ML](#fetch-seed-data-from-open-ml)
  - [Configure credentials](#configure-credentials-here)
- [Demonstrations](#demonstrations)
  - [Measure Brainome (required)](#measure-brainome-required)
  - [Running open_ml_experiment on sagemaker](#running-open_ml_experiment-on-sagemaker)
  - [Running open_ml_experiment on azure](#running-open_ml_experiment-on-azure)
  - [Running open_ml_experiment on google tables](#running-open_ml_experiment-on-google-tables)

## Methodology
We selected 21 binary classification datasets from OpenML. They are a representative subset of the 100 binary classification datasets originally selected by Capital One and the University of Illinois Urbana-Champaign in their 2019 paper
([“Towards Automated Machine Learning: Evaluation and Comparison of AutoML Approaches and Tools“](https://arxiv.org/abs/1908.05557)) by Ahn Truong. This paper attempted to compare the various AutoML platforms available at the time.

Our benchmark recorded five key performance metrics: test accuracy, F1 score, training speed, prediction (or inference) speed and model size.

The four AutoML systems benchmarked are:

    Brainome data compiler (version 1.007) (Brainome)
    
    Google Cloud AutoML Tables / Vertex AI (GCML)
    
    Amazon SageMaker (SageMaker)
    
    Microsoft Azure Machine Learning (AzureML)

There are no “version” numbers for GCML, SageMaker and AzureML but all tests were conducted between December 2021 and January 2022. 

For consistency, all tests were automated via scripting using the respective API of each system. 

Each dataset was split into a training dataset (70%) used exclusively for training and validation of the model and a held back test dataset (30%), used to compute the accuracy of the model. We used Brainome to clean the raw data files downloaded from OpenML to convert all non-numeric data into numbers and fill in any missing values before submitting to all 4 AutoML systems. This was necessary to ensure that all platforms were using the same exact starting point for training.

In order to protect our budget from run away AutoML training charges, model building was limited to 1 hour or the Brainome run which ever was longer. 

Brainome was run on a single EC2 m5.2xlarge. SageMaker and AzureML were run on similar hardware platforms equivalent (the suggested default for SageMaker). We did not have control of GCML’s hardware. 

## Setup
### Cloning Source

```bash
echo "clone source"
git clone git@github.com:brainome/automl-benchmarks.git
sudo apt install unzip
```
## 4 Separate Virtual Environments
Because the big three dependencies do not play well together.

### Setup venv-brainome for Brainome
```bash
cd automl-benchmarks
python3 -m venv venv-brainome
source venv-brainome/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
echo "brainome key required for large files"
brainome login
exit
```
### Setup venv-azure for Azure AutoML
Provision the appropriate resources and limits here

https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources
```bash
cd automl-benchmarks
python3 -m venv venv-azure
source venv-azure/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-azure.txt
echo "create azure resources"
exit
```
### Setup venv-sagemaker for AWS
```bash
echo "install aws cli e.g. python3 -m pip install awscli"
cd automl-benchmarks
python3 -m venv venv-sagemaker
source venv-sagemaker/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-sagemaker.txt
exit
```
### Setup venv-tables for Google Cloud Platform
Google Cloud SDK setup 

https://cloud.google.com/vertex-ai/docs/start/cloud-environment
```bash
echo "setup google vertex env at "
cd automl-benchmarks
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-374.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-sdk-374.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
exit
cd automl-benchmarks
./google-cloud-sdk/bin/gcloud init

python3 -m venv venv-tables
source venv-tables/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-tables.txt
exit
```
## Fetch seed data from Open ML
```bash
echo 'downloading data from open ml into ./data'
python3 open_ml_download_data.py data/
```

## Configure credentials here
```bash
echo "settings required for install three clouds"
vim helpers/user_variable.py
```
```python
credentials = {
    # ##### AWS ######
    # REQUIRES "aws configure"
    # sagemaker role
    "sagemaker_role": "arn:aws:iam::XXXXXXXXXXX:role/role_sagemaker",
    # sagemaker instance type
    "instance_type": "ml.m5.2xlarge",
    # S3 bucket name
    "bucket_name": "my_s3_bucket",			
    # ####### AZURE Workspace params ##########
    # see https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources
    # In the terminal, you may be asked to sign in to authenticate. Copy the code and follow the link to complete this step.
    "workspace_name": 'workspace',
    "subscription_id": 'xxxxxxxxxxxxxxxxxxxxxxxxxx',
    "resource_group": 'groups',
    # ####### AZURE COMPUTE CLUSTER ##############
    "CPU_CLUSTER_NAME": "cpu-cluster-4",
    "vm_size": "Standard_DS12_v2",
    "max_nodes": "4",
    # ######### GOOGLE TABLES ########
    "BUCKET_NAME": "bucket-name",
    "COMPUTE_REGION": 'us-central1',
    "PROJECT_ID": "project-id-1234",
}
```
# Demonstrations
## Measure Brainome (required)
```bash
source venv-brainome/bin/activate
python3 open_ml_brainome_wrapper.py
python3 build_btc_table.py
```

## Running open_ml_experiment on sagemaker 
```bash
source venv-sagemaker/bin/activate
python3 open_ml_experiement.py sagemaker
```

## Running open_ml_experiment on azure 
```bash
source venv-azure/bin/activate
python3 open_ml_experiement.py azure
```

## Running open_ml_experiment on google tables 
```bash
source venv-tables/bin/activate
python3 open_ml_experiement.py tables
```
