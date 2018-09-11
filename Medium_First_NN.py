import numpy as np

class NeuralNetwork():
  def __init__(self):
    self.synapse_weights = np.random.rand(4,1)

  def sigmoid(self, x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid

  def sigmoid_deriv(self, x):
    sigmoid_deriv = np.exp(-x)/((1 + np.exp(-x))**2)
    return sigmoid_deriv

  def train(self, inputs, expected_outputs, its):
    for iteration in range(its):
        guessed_outputs = self.results(inputs)
        error = expected_outputs - guessed_outputs
        slope = self.sigmoid_deriv(guessed_outputs)
        error = error * slope
        self.synapse_weights += np.dot(ts_inputs.T, error)

  def results(self, inputs):
    inputs = inputs.astype(float)
    output = self.sigmoid(np.dot(inputs, self.synapse_weights))
    return output

if __name__ == "__main__":

  ts_inputs = np.array([[0,0,1,0],[1,1,1,0],[1,0,1,1],[0,1,1,1],[0,1,0,1],[1,1,1,1],[0,0,0,0]])
  ts_outputs = np.array([[0,1,1,0,0,1,0]]).T

  test_data = np.array([0,1,1,0])
  neural_network = NeuralNetwork()
  neural_network.train(ts_inputs, ts_outputs, 500)
  print(neural_network.results(test_data))
