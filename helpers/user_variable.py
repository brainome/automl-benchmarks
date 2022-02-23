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


class UndefinedUserVariable(Exception):

	def __init__(self, line_number, script):
		self.line_number = line_number
		self.script = script

	def __str__(self):
		return f'At line {self.line_number} of {self.script} there is a variable that must be defined by the user.'


class UserDefinedVariable:

	creds = {
		# sagemaker role
		"19": "arn:aws:iam::024158331100:role/role_sagemaker"
	}

	def __init__(self, key, script):
		try:
			return self.creds[key]
		except:
			raise UndefinedUserVariable(key, script)