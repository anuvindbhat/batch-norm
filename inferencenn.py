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

class ReLU():
	def transform(self, inputs):
		result = inputs
		result[result < 0] = 0.0
		return result
	def grad(self, outputs):
		result = outputs
		result[result <= 0] = 0.0
		result[result > 0] = 1.0
		return result

class Layer():
	def __init__(self, inpSize, numNodes, actFunc, eta, batch_size):
		self.inpSize = inpSize
		self.size = numNodes
		self.weightMatrix = np.random.randn(inpSize, numNodes)
		self.actFunc = actFunc
		self.bias = np.random.randn(1, numNodes)
		self.eta = eta
		self.batch_size = batch_size
		#caching the node input for the backward pass
		self.inp = np.zeros((batch_size, inpSize))
		self.out = np.zeros((batch_size, numNodes))

	def forwardPass(self, inp):
		self.inp = inp
		net = np.dot(inp, self.weightMatrix) + self.bias
		self.out = self.actFunc.transform(net)
		return self.out

	def forwardPassTest(self, inp):
		net = np.dot(inp, self.weightMatrix) + self.bias
		return self.actFunc.transform(net)

	def backwardPass(self, error):
		backprop_error = error * self.actFunc.grad(self.out)
		next_error = np.dot(backprop_error, self.weightMatrix.T)
		self.weightMatrix += self.eta * np.dot(self.inp.T, backprop_error)
		self.bias += self.eta * np.dot(np.ones((self.inp.shape[0], 1)).T, backprop_error)
		return next_error

class NeuralNet():
	ETA = 0.03
	def __init__(self, inputSize, layerSizes, activationFunctions, batch_size):
		self.layers = []
		self.batch_size = batch_size
		for i in range(len(layerSizes)):
			if i == 0:
				self.layers.append(Layer(inputSize, layerSizes[0], activationFunctions[0], self.ETA, batch_size))
			else:
				self.layers.append(Layer(layerSizes[i - 1], layerSizes[i], activationFunctions[i], self.ETA, batch_size))

	def forwardPass(self, inp):
		out = inp
		for layer in self.layers:
			out = layer.forwardPass(out)
		return out

	def forwardPassTest(self, inp):
		out = inp
		for layer in self.layers:
			out = layer.forwardPassTest(out)
		return out

	def error(self, out, target):
		return target - out

	def backwardPass(self, error):
		backprop_error = error
		for layer in reversed(self.layers):
			backprop_error = layer.backwardPass(backprop_error)

	def batch_train(self, epochs, inp, target, test_inp, test_target):
		num_batches = int(inp.shape[0] / self.batch_size)
		epoch_list = []
		loss_list = []
		test_accuracy = []
		for i in range(epochs):
			loss = 0.0
			randomize = np.arange(len(inp))
			np.random.shuffle(randomize)
			inp = inp[randomize]
			target = target[randomize]
			for j in range(num_batches):
				out = self.forwardPass(inp[j*self.batch_size:(j+1)*self.batch_size, :])
				error = self.error(out, target[j*self.batch_size:(j+1)*self.batch_size, :])
				loss += 0.5 * np.sum(error ** 2)
				self.backwardPass(error / self.batch_size)
			epoch_list.append(i + 1)
			loss_list.append(loss)
			test_accuracy.append(self.test(test_inp, test_target))
			print('Epoch #%d: Loss %f, Test Accuracy %f' % (i + 1, loss, test_accuracy[-1]))
		return epoch_list, loss_list, test_accuracy

	def test(self, inp, target):
		out = self.forwardPassTest(inp)
		selectedIndices = np.argmax(out, axis=1)
		targetIndices = np.argmax(target, axis=1)
		count = 0
		for i in range(len(selectedIndices)):
			if selectedIndices[i] == targetIndices[i]:
				count += 1
		return count / len(selectedIndices)
