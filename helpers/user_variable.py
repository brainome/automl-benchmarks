
class UndefinedUserVariable(Exception):

	def __init__(self, line_number, script):
		self.line_number = line_number
		self.script = script

	def __str__(self):
		return f'At line {self.line_number} of {self.script} there is a variable that must be defined by the user.'

class UserDefinedVariable:

	def __init__(self, line_number, script):
		raise UndefinedUserVariable(line_number, script)