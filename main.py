import numpy
import numpy as np


def LinearActivation(x):
    return x


def dLinearActivation(x):
    return 1


def ReLU(x):
    return max(0, x)


def dReLU(x):
    return max(0, 1)


def Softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dSigmoid(x):
    s = Sigmoid(x)
    return s * (1 - s)


def dTanH(x):
    return 1 - np.tanh(x)**2


class Layer:
    def __init__(self, layerType, size, activation):
        self.layerType = layerType
        self.size = size
        self.output = np.zeros(size)
        self.dOutput = np.zeros(size)
        self.gradient = None
        self.prev = None
        self.next = None

        if activation == "Linear":
            self.activation = LinearActivation
            self.dActivation = dLinearActivation
        elif activation == "ReLU":
            self.activation = ReLU
            self.dActivation = dReLU
        elif activation == "TanH":
            self.activation = np.tanh
            self.dActivation = dTanH
        elif activation == "Softmax":
            self.activation = Softmax
            self.dActivation = NotImplementedError
        else:
            raise ValueError("[ERROR] Unknown activation function")

    def feedForward(self):
        pass

    def calculateGradient(self, nGradient=None):
        pass

    def updateParameters(self, n, learningRate):
        pass

    def setup_(self, prev, nextL):
        self.prev = prev
        self.next = nextL


class InputLayer(Layer):
    def __init__(self, size):
        super().__init__("InputLayer", size, "Linear")


class FFLayer(Layer):
    def __init__(self, size, activation="Linear"):
        super().__init__("FFLayer", size, activation)

        self.weights = None
        self.biases = np.zeros(self.size)
        self.gradient = np.zeros(self.size)

    def feedForward(self):
        self.output = self.weights.dot(self.prev.output) + self.biases
        self.dOutput = self.dActivation(self.output)
        self.output = self.activation(self.output)

    def calculateGradient(self, nGradient=None):
        if nGradient is not None:
            self.gradient += self.next.weights.T.dot(nGradient) * self.dOutput
        else:
            self.gradient += self.next.weights.T.dot(self.next.gradient) * self.dOutput

    def updateParameters(self, n, learningRate):
        # Average the gradients
        self.gradient /= n
        self.weights -= learningRate * self.gradient.dot(self.prev.output.T)
        self.biases -= learningRate * self.gradient

    def setup_(self, prev, nextL):
        super().setup_(prev, nextL)

        self.weights = np.random.rand(self.size, self.prev.size)


class LSTMLayer(Layer):
    def __init__(self, size, activation="Linear"):
        super().__init__("FFLayer", size, activation)
        self.fWeights = None
        self.iWeights = None
        self.cWeights = None
        self.oWeights = None

        self.fBiases = np.zeros(self.size)
        self.iBiases = np.zeros(self.size)
        self.cBiases = np.zeros(self.size)
        self.oBiases = np.zeros(self.size)
        self.states = np.zeros(size)

        self.gradient = np.array((size, 4))

    def feedForward(self):
        vt = np.concatenate((self.states, self.prev.output))

        ft = Sigmoid(self.fWeights.dot(vt) + self.fBiases)
        it = Sigmoid(self.iWeights.dot(vt) + self.iBiases)
        ct = np.tanh(self.cWeights.dot(vt) + self.cBiases)
        ot = Sigmoid(self.oWeights.dot(vt) + self.oBiases)

        self.output = ft * self.output + it * ct
        self.states = ot * np.tanh(self.output)

    def calculateGradient(self, nGradient=None):
        # self.gradient = self.weights.T.dot(self.next.gradient) * self.dOutput
        return self.gradient

    def setup_(self, prev, nextL):
        super().setup_(prev, nextL)

        self.fWeights = np.random.rand(self.size, self.size + self.prev.size)
        self.iWeights = np.random.rand(self.size, self.size + self.prev.size)
        self.cWeights = np.random.rand(self.size, self.size + self.prev.size)
        self.oWeights = np.random.rand(self.size, self.size + self.prev.size)


class AI:
    def __init__(self):
        self.layers = [InputLayer(3),
                       LSTMLayer(4),
                       FFLayer(2, activation="TanH")]

        self.setupLayers()
        self.learningRate = 0.001

    def setupLayers(self):
        if len(self.layers) < 2:
            raise ValueError(f"[ERROR] At least 3 layers are required")

        if self.layers[0].layerType != "InputLayer":
            raise ValueError(f"[ERROR] First layer isn't InputLayer")

        self.layers[0].setup_(None, self.layers[1])
        for i in range(1, len(self.layers) - 1):
            self.layers[i].setup_(self.layers[i - 1], self.layers[i + 1])
        self.layers[-1].setup_(self.layers[-2], None)

    def feedForward(self, inputState):
        if inputState.shape != self.layers[0].output.shape:
            raise ValueError(
                f"[ERROR] Feed-forward input's shape is not {self.layers[0].output.shape} but {inputState.shape}")

        self.layers[0].output = inputState
        for i in range(1, len(self.layers)):
            self.layers[i].feedForward()

    def train(self, dataset, epochs=1):
        # For each epoch
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # For each sentence
            for sentence in dataset:

                # For each word / timestep
                for inputState, actual in sentence:
                    self.feedForward(inputState)

                    lossDerivative = (2 / len(self.layers)) * (self.layers[-1].output - actual)
                    # errorL = lossDerivative * self.layers[-1].dActivation(self.layers[-1].zNeurons)
                    errorL = lossDerivative * self.layers[-1].dOutput

                    # L-1 .. 0
                    for i in range(len(self.layers) - 2, -1, -1):
                        layer = self.layers[i]
                        layer.calculateGradient()

                    # self.layers[-2].weights -= 0.001 * errorL * self.layers[-2].neurons

                # Sum gradients for each layer
                # gradients = np.sum(gradients, axis=0)

            for layer in self.layers:
                layer.updateParameters(sum([len(d[1]) for d in dataset]), self.learningRate)


ai = AI()

# ai.feedForward(np.ones(ai.layers[0].size))
# for layer in ai.layers:
#     if layer.layerType == "LSTMLayer":
#         print(layer.states)
#         print(layer.output)
#     else:
#         print(layer.output)
#     print()

dataset = [
    [
        [np.array([1, 2, 3]), np.array([0, 0.5])],
        [np.array([4, 1, 2]), np.array([1, 0])]
    ],
    [
        [np.array([5, 2, 1]), np.array([1, 0.0])],
        [np.array([0, 1, 2]), np.array([1, 0.0])]
    ]
]
ai.train(dataset, epochs=1)

# print(a.neurons)
# print(a.weights.dot(a.neurons))
