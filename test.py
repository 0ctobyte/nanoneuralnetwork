import sklearn.datasets, numpy as np, NanoNeuralNetwork as nnn, matplotlib.pyplot as plt

n_samples = 100000
n_features = 20
n_classes = 1
train_pct = 0.98
dev_pct = 0.01
test_pct = 0.01
iterations = 2500
layers = [n_features, 5, 3, n_classes]
random_state = 1

samples, labels = sklearn.datasets.make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=random_state)
labels = np.reshape(labels, (labels.shape[0], 1))

n_train_samples = int(n_samples * train_pct)
n_dev_samples = int(n_samples * dev_pct)
n_test_samples = int(n_samples * test_pct)

dev_sample_start = n_train_samples
dev_sample_end = dev_sample_start + n_dev_samples
test_sample_start = dev_sample_end

train_samples, train_labels = samples[:n_train_samples], labels[:n_train_samples]
dev_samples, dev_labels = samples[dev_sample_start:dev_sample_end], labels[dev_sample_start:dev_sample_end]
test_samples, test_labels = samples[test_sample_start:], labels[test_sample_start:]

model = nnn.NanoNeuralNetwork(layers=layers)
costs = model.train(train_samples, train_labels, iterations=iterations)

fig, ax = plt.subplots()
ax.plot(costs)
ax.set_title("Cost vs Iteration while Training the Neural Network Model")
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
ax.set_ylim([0, 1])
plt.show()
