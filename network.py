import numpy as np

X = np.array((
  [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], # A
  [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0], # B
  [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0], # C
  [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0], # D
  [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1], # E
  [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], # F
))


y = np.zeros((6, 6))
np.fill_diagonal(y, 1)


class LetterRecognizer(object):
  def __init__(self, learning_rate, input_size, output_size):
    self.loss = []
    self.learning_rate = learning_rate
    hidden_size = int((input_size + output_size) / 2)
    self.weights1 = np.random.randn(input_size, hidden_size)
    self.weights2 = np.random.randn(hidden_size, output_size)


  def forward(self, X):
    hidden_layer_sum = np.dot(X, self.weights1)
    self.hidden_layer_result = self.activation(hidden_layer_sum)
    output_layer_sym = np.dot(self.hidden_layer_result, self.weights2)
    output_layer_result = self.activation(output_layer_sym)
    return output_layer_result


  def activation(self, s):
    return 1 / (1 + np.exp(-s))


  def activation_derivative(self, s):
    return s * (1 - s)


  def backward(self, X, y, o):
    output_delta = (y - o) * self.activation_derivative(o)
    hidden_delta = output_delta.dot(self.weights2.T) * self.activation_derivative(self.hidden_layer_result)

    self.weights1 += X.T.dot(hidden_delta) * self.learning_rate
    self.weights2 += self.hidden_layer_result.T.dot(output_delta) * self.learning_rate


  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)


LR = LetterRecognizer(0.5, 24, 6)
for i in range(100):
  LR.train(X, y)

