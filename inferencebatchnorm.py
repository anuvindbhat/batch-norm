import numpy as np

# Used to prevent normalized X
# from blowing up (numerical stability)
EPSILON = 1e-12

# x is (m x d) where m is mini-batch size
# and d is the number of dimensions

def forward_batch(gamma, beta, x):
	# Mean of the mini-batch, mu
	mu = np.mean(x, axis=0)

	# Variance of the mini-batch, sigma^2
	var = np.var(x, axis=0)
	std_inv = 1.0 / np.sqrt(var + EPSILON)

	# The normalized input, x_hat
	x_hat = (x - mu) * std_inv

	# Batch normalizing (affine) transformation
	y = gamma * x_hat + beta

	backprop_stuff = (gamma, std_inv, x_hat)

	return y, mu, var, backprop_stuff

def forward_batch_test(gamma, beta, x, mu, var):
	std_inv = 1.0 / np.sqrt(var + EPSILON)

	# The normalized input, x_hat
	x_hat = (x - mu) * std_inv

	# Batch normalizing (affine) transformation
	y = gamma * x_hat + beta

	return y

def backward_batch(dy, backprop_stuff):
	# m: mini-batch size
	# d: number of dimensions
	m, d = dy.shape
	gamma, std_inv, x_hat = backprop_stuff

	# Equation 1 in paper
	dx_hat = dy * gamma

	# Equation 2, 3, 4 in paper
	# Equation 2 was rederived as the one in the paper was not simplified
	dx = std_inv * (dx_hat - np.mean(dx_hat, axis=0) - x_hat * np.mean(dx_hat * x_hat, axis=0))
	dgamma = np.sum(dy * x_hat, axis=0)
	dbeta = np.sum(dy, axis=0)

	return dx, dgamma, dbeta

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
		self.gamma = np.random.randn(1, numNodes)
		self.beta = np.random.randn(1, numNodes)
		#caching the node input for the backward pass
		self.inp = np.zeros((batch_size, inpSize))
		self.out = np.zeros((batch_size, numNodes))
		self.backprop_stuff = None
		self.mean_training = np.zeros((1, numNodes))
		self.var_training = np.zeros((1, numNodes))

	def forwardPass(self, inp):
		self.inp = inp
		net = np.dot(inp, self.weightMatrix) + self.bias
		y, mu, var, self.backprop_stuff = forward_batch(self.gamma, self.beta, net)
		self.mean_training = 0.9 * self.mean_training + 0.1 * mu
		self.var_training = 0.9 * self.var_training + 0.1 * var
		self.out = self.actFunc.transform(y)
		return self.out

	def forwardPassTest(self, inp):
		net = np.dot(inp, self.weightMatrix) + self.bias
		y = forward_batch_test(self.gamma, self.beta, net, self.mean_training, self.var_training)
		return self.actFunc.transform(y)

	def backwardPass(self, error):
		backprop_error = error * self.actFunc.grad(self.out)
		backprop_error, dgamma, dbeta = backward_batch(backprop_error, self.backprop_stuff)
		next_error = np.dot(backprop_error, self.weightMatrix.T)
		self.weightMatrix += self.eta * np.dot(self.inp.T, backprop_error)
		self.bias += self.eta * np.dot(np.ones((self.inp.shape[0], 1)).T, backprop_error)
		self.gamma += dgamma
		self.beta += dbeta
		return next_error

class NeuralNet():
	ETA = 0.9
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
