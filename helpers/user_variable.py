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
		"sagemaker_role": "arn:aws:iam::024158331100:role/role_sagemaker",		# TODO Obfuscate
		# sagemaker instance type
		"instance_type": "ml.m5.2xlarge",
		# S3 bucket name
		"bucket_name": "download.brainome.ai",									# TODO Obfuscate
		# ####### AZURE Workspace params ##########
		"workspace_name": 'brainome',											# TODO Obfuscate
		"subscription_id": 'b245cbde-7433-4f70-b19a-9c812b627b1b',				# TODO Obfuscate
		"resource_group": 'brainome',											# TODO Obfuscate
		# ####### AZURE COMPUTE CLUSTER ##############
		"CPU_CLUSTER_NAME": "cpu-cluster-4",
		"vm_size": "Standard_DS12_v2",
		"max_nodes": "4",
		# ######### GOOGLE TABLES ########
		"BUCKET_NAME": "brainome-automl-central",								# TODO Obfuscate
		"COMPUTE_REGION": 'us-central1',
		"PROJECT_ID": "fifth-glazing-334722",									# TODO Obfuscate
	}

	@classmethod
	def get(cls, key):
		try:
			return cls.credentials[key]
		except KeyError:
			raise PermissionError(f"ERROR Mmissing credentials for {key} in user_variables.py")
