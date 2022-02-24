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

## Installation / Setup

```bash
echo "clone source"
git clone git@github.com:brainome/automl-benchmarks.git
sudo apt install unzip

echo "From a new terminal session for each"
echo "create/activate FOUR virtual envs - one per vendor"
echo "################## BRAINOME ##################"
cd automl-benchmarks
python3 -m venv venv-brainome
source venv-brainome/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
echo "brainome key required for large files"
brainome login
exit

echo "################## AZURE ##################"
cd automl-benchmarks
python3 -m venv venv-azure
source venv-azure/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-azure.txt
echo "create azure resources"
exit

echo "################## SAGEMAKER ##################"
echo "\ninstall aws cli e.g. python3 -m pip install awscli"
cd automl-benchmarks
python3 -m venv venv-sagemaker
source venv-sagemaker/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-sagemaker.txt
exit

echo "#################### GOOGLE VERTEX / TABLES ############"
echo "\nsetup google vertex env at https://cloud.google.com/vertex-ai/docs/start/cloud-environment"
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-374.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-sdk-374.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
exit
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


## Initialization
```bash
echo "settings required for install three clouds"
vim helpers/user_variable.py
```

## Demonstrations
### Usage
python3 open_ml_experiement.py <tool_name> <suite_tsv_file> <data_dir>

### Running brainome benchmark
python3 open_ml_brainome_wrapper.py test-suites/open_ml_select.tsv /Dropbox/OpenML-Data/

### Running open_ml_experiment
```bash
python3 open_ml_experiement.py sagemaker test-suites/open_ml_select.tsv /Dropbox/OpenML-Data/
```

### Results and Predictions
	RESULT_DIR = f"{tool}-runs"
	PREDICTIONS_DIR = f"{tool}-predictions"