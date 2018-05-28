import numpy as np
import matplotlib.pyplot as plt
import mnist
import inferencenn as NN
import inferencebatchnorm as BNN

def get_data_set():
	train_images = mnist.train_images()
	train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
	train_labels = mnist.train_labels()
	index_offset_train = np.arange(train_labels.shape[0]) * 10
	one_hot_train = np.zeros((train_labels.shape[0], 10))
	one_hot_train.flat[index_offset_train + train_labels.ravel()] = 1

	test_images = mnist.test_images()
	test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
	test_labels = mnist.test_labels()
	index_offset_test = np.arange(test_labels.shape[0]) * 10
	one_hot_test = np.zeros((test_labels.shape[0], 10))
	one_hot_test.flat[index_offset_test + test_labels.ravel()] = 1

	return train_images, one_hot_train, test_images, one_hot_test

train_data, train_labels, test_data, test_labels = get_data_set()

# Number of samples should be divisible by batch size
NNet = NN.NeuralNet(train_data.shape[1], [60, 60, train_labels.shape[1]], [NN.TanSigmoid(), NN.TanSigmoid(), NN.LogSigmoid()], 30)
BNNet = BNN.NeuralNet(train_data.shape[1], [60, 60, train_labels.shape[1]], [BNN.TanSigmoid(), BNN.TanSigmoid(), BNN.LogSigmoid()], 30)

total_epochs = 10
epochsN, lossN, testN = NNet.batch_train(total_epochs, train_data, train_labels, test_data, test_labels)
epochsB, lossB, testB = BNNet.batch_train(total_epochs, train_data, train_labels, test_data, test_labels)
plt.plot(epochsN, lossN)
plt.plot(epochsB, lossB)
plt.ylabel('Loss (MSE)')
plt.xlabel('Epochs')
plt.legend(('Vanilla Batch', 'Batch Normalization'))
plt.savefig('loss.png', bbox_inches='tight')
plt.close()
plt.plot(epochsN, testN)
plt.plot(epochsB, testB)
plt.ylabel('Test Accuracy')
plt.xlabel('Epochs')
plt.legend(('Vanilla Batch', 'Batch Normalization'))
plt.savefig('test.png', bbox_inches='tight')
plt.close()
