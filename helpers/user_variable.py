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


class UserDefinedVariable:

	credentials = {
		# ##### AWS ######
		# REQUIRES "aws configure"
		# sagemaker role
# 		"sagemaker_role": "arn:aws:iam::XXXXXXXXXXXXX:role/role_sagemaker",
		# sagemaker instance type
		"instance_type": "ml.m5.2xlarge",
		# S3 bucket name
# 		"bucket_name": "XXXXXXXXXXXX",
		# ####### AZURE Workspace params ##########
		# see https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources
		# In the terminal, you may be asked to sign in to authenticate. Copy the code and follow the link to complete this step.
# 		"workspace_name": 'XXXXXXXXXXXXXXX',
# 		"subscription_id": 'XXXXXXXXXXXXXXXXX',
# 		"resource_group": 'XXXXXXXXXXXXX',
		# ####### AZURE COMPUTE CLUSTER ##############
		"CPU_CLUSTER_NAME": "cpu-cluster-4",
		"vm_size": "Standard_DS12_v2",
		"max_nodes": "4",
		# ######### GOOGLE TABLES ########
# 		"BUCKET_NAME": "bucket-name",
# 		"COMPUTE_REGION": 'us-central1',
# 		"PROJECT_ID": "project-id-1234",
	}

	@classmethod
	def get(cls, key):
		try:
			return cls.credentials[key]
		except KeyError:
			raise PermissionError(f"ERROR Mmissing credentials for {key} in user_variables.py")
