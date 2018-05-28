import numpy as np

class LogSigmoid():
	def transform(self, inputs):
		result = np.empty(inputs.shape)
		t = np.exp(-inputs[inputs >= 0])
		result[inputs >= 0] = 1.0 / (1.0 + t)
		t = np.exp(inputs[inputs < 0])
		result[inputs < 0] = t / (1.0 + t)
		return result
	def grad(self, outputs):
		return outputs * (1 - outputs)

class TanSigmoid():
	def transform(self, inputs):
		result = np.empty(inputs.shape)
		t = np.exp(-2.0 * inputs[inputs >= 0])
		result[inputs >= 0] = (1.0 - t) / (1.0 + t)
		t = np.exp(2.0 * inputs[inputs < 0])
		result[inputs < 0] = (t - 1.0) / (t + 1.0)
		return result
	def grad(self, outputs):
		return 1.0 - outputs * outputs

l = LogSigmoid()
t = TanSigmoid()

test = np.arange(-2, 2.1, 0.1)
ltest = l.transform(test)
ttest = t.transform(test)
for x, y, z in zip(test, ltest, ttest):
	print('%f\t%f\t%f' % (x, y, z))
