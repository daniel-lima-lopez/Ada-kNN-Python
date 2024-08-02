# Ada-kNN-Python
This is a python implementation of the Ada-kNN classifier, proposed by Mullick et al.

## Classifier description
The classifier is an extension of kNN, in which a strategy based on Multi-layer perceptron (MLP) is proposed to automate the choice of the parameter k, for each instance to be classified.

The operation of Ada-kNN is described bellow:
- First, for each instance $x_i$ in the training set, a series of experiments are performed to identify the $k$ values which correctly classify $x_i$ with kNN.
- With this information, a MLP architecture is trained to predict the most appropriate $k$ value for classifying a given instance based on their attribute values.
- Once the neural network has been trained, for each instance to be classified, the most appropriate value of $k$ is predicted with this network, then a conventional kNN classifier performs a prediction with this value.