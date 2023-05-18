import numpy
import numpy as np


def NoneActivation(x):
    return x


def dNoneActivation(x):
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


class Layer:
    def __init__(self, layerType, size):
        self.layerType = layerType
        self.size = size
        self.output = np.zeros(size)
        self.prev = None
        self.next = None

    def feedForward(self):
        pass

    def calculateGradient(self):
        pass

    def setup_(self, prev, nextL):
        self.prev = prev
        self.next = nextL


class ActivationLayer(Layer):
    def __init__(self, activation):
        super().__init__("ActivationLayer", 0)
        self.dOutput = None

        if activation == "none":
            self.activation = NoneActivation
            self.dActivation = dNoneActivation
        elif activation == "ReLU":
            self.activation = ReLU
            self.dActivation = dReLU
        elif activation == "Softmax":
            self.activation = Softmax
            self.dActivation = NotImplementedError
        else:
            raise ValueError("[ERROR] Unknown activation function")

    def feedForward(self):
        self.output = self.activation(self.prev.output)
        self.dOutput = self.dActivation(self.prev.output)

    def setup_(self, prev, nextL):
        super().setup_(prev, nextL)

        self.size = prev.size
        self.output = np.zeros(prev.size)
        self.dOutput = np.zeros(prev.size)


class InputLayer(Layer):
    def __init__(self, size):
        super().__init__("InputLayer", size)


class FFLayer(Layer):
    def __init__(self, size):
        super().__init__("FFLayer", size)

        self.weights = None
        self.biases = np.zeros(self.size)

    def feedForward(self):
        self.output = self.weights.dot(self.prev.output) + self.biases

    def calculateGradient(self):
        pass

    def setup_(self, prev, nextL):
        super().setup_(prev, nextL)

        self.weights = np.random.rand(self.size, self.prev.size)


class LSTMLayer(Layer):
    def __init__(self, size):
        super().__init__("FFLayer", size)
        self.fWeights = None
        self.iWeights = None
        self.cWeights = None
        self.oWeights = None

        self.fBiases = np.zeros(self.size)
        self.iBiases = np.zeros(self.size)
        self.cBiases = np.zeros(self.size)
        self.oBiases = np.zeros(self.size)
        self.states = np.zeros(size)

    def feedForward(self):
        vt = np.concatenate((self.states, self.prev.output))

        ft = Sigmoid(self.fWeights.dot(vt) + self.fBiases)
        it = Sigmoid(self.iWeights.dot(vt) + self.iBiases)
        ct = np.tanh(self.cWeights.dot(vt) + self.cBiases)
        ot = Sigmoid(self.oWeights.dot(vt) + self.oBiases)

        self.output = ft * self.output + it * ct
        self.states = ot * np.tanh(self.output)

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
                       FFLayer(6),
                       ActivationLayer("Softmax")]

        self.setupLayers()
        self.learningRate = 0.001

    def setupLayers(self):
        if len(self.layers) < 2:
            raise ValueError(f"[ERROR] At least 3 layers are required")

        if self.layers[0].layerType != "InputLayer":
            raise ValueError(f"[ERROR] First layer isn't InputLayer")

        self.layers[0].setup_(None, self.layers[1])
        for i in range(1, len(self.layers)-1):
            self.layers[i].setup_(self.layers[i - 1], self.layers[i + 1])
        self.layers[-1].setup_(self.layers[-2], None)

    def feedForward(self, inputState):
        if inputState.shape != self.layers[0].output.shape:
            raise ValueError(f"[ERROR] Feed-forward input's shape is not {self.layers[0].output.shape} but {inputState.shape}")

        self.layers[0].output = inputState
        for i in range(1, len(self.layers)):
            self.layers[i].feedForward()

    def gradientDescent(self, actual):
        lossDerivative = (2/len(self.layers)) * (self.layers[-1].output - actual)
        #errorL = lossDerivative * self.layers[-1].dActivation(self.layers[-1].zNeurons)
        errorL = lossDerivative * self.layers[-1].dOutput

        # L-1 .. 0
        for i in range(self.layerCount-2, -1, -1):
            layer = self.layers[i]
            errorl = layer.dActivation()
        #self.layers[-2].weights -= 0.001 * errorL * self.layers[-2].neurons


ai = AI()

ai.feedForward(np.ones(ai.layers[0].size))
for layer in ai.layers:
    if layer.layerType == "LSTMLayer":
        print(layer.states)
        print(layer.output)
    else:
        print(layer.output)
    print()
#ai.gradientDescent(np.array([1, 2, 3, 4]))

# print(a.neurons)
# print(a.weights.dot(a.neurons))

