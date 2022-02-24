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
ML benchmarks comparing brainome, google, sage maker, and azure engines

## SETUP
### Cloning Source

```bash
echo "clone source"
git clone git@github.com:brainome/automl-benchmarks.git
sudo apt install unzip
```
### From a new terminal session for each, create/activate FOUR virtual envs

### Setup venv-brainome
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
### Setup venv-azure
```bash
cd automl-benchmarks
python3 -m venv venv-azure
source venv-azure/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-azure.txt
echo "create azure resources"
exit
```
### Setup venv-sagemaker
```bash
echo "install aws cli e.g. python3 -m pip install awscli"
cd automl-benchmarks
python3 -m venv venv-sagemaker
source venv-sagemaker/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-sagemaker.txt
exit
```
### Setup venv-tables
```bash
echo "setup google vertex env at https://cloud.google.com/vertex-ai/docs/start/cloud-environment"
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
echo 'downloading data from open ml into /Dropbox/Open_ML-Data/'
mkdir -f "/Dropbox/Open_ML-Data/"
python3 opem_ml_download_data.py
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
python3 open_ml_brainome_wrapper.py test-suites/open_ml_select.tsv data/
```

## Running open_ml_experiment on sagemaker 
```bash
source venv-sagemaker/bin/activate
python3 open_ml_experiement.py sagemaker test-suites/open_ml_select.tsv data/
```

## Running open_ml_experiment on azure 
```bash
source venv-azure/bin/activate
python3 open_ml_experiement.py azure test-suites/open_ml_select.tsv data/
```

## Running open_ml_experiment on google tables 
```bash
source venv-tables/bin/activate
python3 open_ml_experiement.py tables test-suites/open_ml_select.tsv data/
```
