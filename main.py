import numpy
import numpy as np


def LinearActivation(x):
    return x


def dLinearActivation(x):
    return 1


def ReLU(x):
    return np.maximum(0, x)


def dReLU(x):
    return np.greater(x, 0).astype(int)


def Softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dSigmoid(x):
    s = Sigmoid(x)
    return s * (1 - s)


def dTanH(x):
    return 1 - np.tanh(x) ** 2


class Layer:
    def __init__(self, layerType, size, activation, gradientCount):
        self.layerType = layerType
        self.size = size
        self.output = np.zeros(size)
        self.dOutput = np.zeros(size)
        self.gradient = None
        self.prev = None
        self.next = None
        self.optimizerFunc = None
        self.gradientCount = gradientCount
        self.optAttrs = {}

        if activation == "Linear":
            self.activation = LinearActivation
            self.dActivation = dLinearActivation
        elif activation == "ReLU":
            self.activation = ReLU
            self.dActivation = dReLU
        elif activation == "TanH":
            self.activation = np.tanh
            self.dActivation = dTanH
        elif activation == "Sigmoid":
            self.activation = Sigmoid
            self.dActivation = dSigmoid
        elif activation == "Softmax":
            self.activation = Softmax
            self.dActivation = NotImplementedError
        else:
            raise ValueError("[ERROR] Unknown activation function")

    def feedForward(self):
        pass

    def calculateGradient(self):
        pass

    def updateParameters(self, n):
        pass

    def reset(self):
        pass

    def setup_(self, prev, nextL, optimizerFunc):
        self.prev = prev
        self.next = nextL
        self.optimizerFunc = optimizerFunc


class InputLayer(Layer):
    def __init__(self, shape):
        super().__init__("InputLayer", np.prod(shape), "Linear", 0)
        self.shape = shape


class FFLayer(Layer):
    def __init__(self, size, activation="Linear"):
        super().__init__("FFLayer", size, activation, 2)

        self.weights = None
        self.biases = np.zeros(self.size)
        self.gradient = np.zeros(self.size)

    def feedForward(self):
        self.output = self.weights.dot(self.prev.output) + self.biases
        self.dOutput = self.dActivation(self.output)
        self.output = self.activation(self.output)

    def calculateGradient(self):
        self.gradient += self.next.weights.T.dot(self.next.gradient) * self.dOutput

    def updateParameters(self, n):
        # Average the gradients
        self.gradient /= n
        self.weights -= self.optimizerFunc(self, 0, self.gradient.dot(self.output[np.newaxis].T))
        self.biases -= self.optimizerFunc(self, 1, self.gradient)

    def setup_(self, prev, nextL, optimizerFunc):
        super().setup_(prev, nextL, optimizerFunc)

        self.weights = np.random.rand(self.size, self.prev.size)


class LSTMLayer(Layer):
    def __init__(self, size, activation="Linear"):
        super().__init__("FFLayer", size, activation, 12)

        self.f1Weights = None
        self.i1Weights = None
        self.c1Weights = None
        self.o1Weights = None

        self.f2Weights = None
        self.i2Weights = None
        self.c2Weights = None
        self.o2Weights = None

        self.fBiases = np.zeros(self.size)
        self.iBiases = np.zeros(self.size)
        self.cBiases = np.zeros(self.size)
        self.oBiases = np.zeros(self.size)

        self.f1WeightsGr = None
        self.i1WeightsGr = None
        self.c1WeightsGr = None
        self.o1WeightsGr = None

        self.f2WeightsGr = None
        self.i2WeightsGr = None
        self.c2WeightsGr = None
        self.o2WeightsGr = None

        self.fBiasesGr = np.zeros(self.size)
        self.iBiasesGr = np.zeros(self.size)
        self.cBiasesGr = np.zeros(self.size)
        self.oBiasesGr = np.zeros(self.size)

        self.zf = None
        self.zi = None
        self.zg = None
        self.zo = None

        self.ft = None
        self.it = None
        self.gt = None
        self.ot = None

        self.states = np.zeros(size)

    def feedForward(self):
        self.zf = self.f1Weights.dot(self.states) + self.f2Weights.dot(self.prev.output) + self.fBiases
        self.ft = Sigmoid(self.zf)

        self.zi = self.i1Weights.dot(self.states) + self.i2Weights.dot(self.prev.output) + self.iBiases
        self.it = Sigmoid(self.zi)

        self.zg = self.c1Weights.dot(self.states) + self.c2Weights.dot(self.prev.output) + self.cBiases
        self.gt = np.tanh(self.zg)

        self.zo = self.o1Weights.dot(self.states) + self.o2Weights.dot(self.prev.output) + self.oBiases
        self.ot = Sigmoid(self.zo)

        self.output = self.ft * self.output + self.it * self.zg
        self.states = self.ot * np.tanh(self.output)

    def calculateGradient(self):
        nGradient = self.next.weights.T.dot(self.next.gradient)

        tmp1 = nGradient * np.tanh(self.output) * dSigmoid(self.zo)
        tmp2 = nGradient * self.ot * dTanH(self.output) * self.output * dSigmoid(self.zf)
        tmp3 = nGradient * self.ot * dTanH(self.output) * self.gt * dSigmoid(self.zi)
        tmp4 = nGradient * self.ot * dTanH(self.output) * self.it * dTanH(self.zg)

        tmp1r = tmp1.reshape((-1, 1))
        tmp2r = tmp2.reshape((-1, 1))
        tmp3r = tmp3.reshape((-1, 1))
        tmp4r = tmp4.reshape((-1, 1))

        prevOutput = self.prev.output.reshape((-1, 1))
        states = self.states.reshape((-1, 1))

        # print(np.dot(prevOutput, tmp1.T))

        # Normal transpose
        # self.prev.output.T

        self.f1WeightsGr = np.dot(tmp1r, states.T)
        self.f2WeightsGr = np.dot(tmp1r, prevOutput.T)
        self.fBiasesGr = tmp1

        self.i1WeightsGr = np.dot(tmp2r, states.T)
        self.i2WeightsGr = np.dot(tmp2r, prevOutput.T)
        self.iBiasesGr = tmp2

        self.c1WeightsGr = np.dot(tmp3r, states.T)
        self.c2WeightsGr = np.dot(tmp3r, prevOutput.T)
        self.cBiasesGr = tmp3

        self.o1WeightsGr = np.dot(tmp4r, states.T)
        self.o2WeightsGr = np.dot(tmp4r, prevOutput.T)
        self.oBiasesGr = tmp4

    def updateParameters(self, n):
        self.f1Weights -= self.optimizerFunc(0, self.f1WeightsGr)
        self.f2Weights -= self.optimizerFunc(1, self.f2WeightsGr)
        self.fBiases -= self.optimizerFunc(2, self.fBiasesGr)

        self.i1Weights -= self.optimizerFunc(3, self.i1WeightsGr)
        self.i2Weights -= self.optimizerFunc(4, self.i2WeightsGr)
        self.iBiases -= self.optimizerFunc(5, self.iBiasesGr)

        self.c1Weights -= self.optimizerFunc(6, self.c1WeightsGr)
        self.c2Weights -= self.optimizerFunc(7, self.c2WeightsGr)
        self.cBiases -= self.optimizerFunc(8, self.cBiasesGr)

        self.o1Weights -= self.optimizerFunc(9, self.o1WeightsGr)
        self.o2Weights -= self.optimizerFunc(10, self.o2WeightsGr)
        self.oBiases -= self.optimizerFunc(11, self.oBiasesGr)

    def reset(self):
        self.states = np.zeros(self.size)

    def setup_(self, prev, nextL, optimizerFunc):
        super().setup_(prev, nextL, optimizerFunc)

        self.f1Weights = np.random.rand(self.size, self.size)
        self.i1Weights = np.random.rand(self.size, self.size)
        self.c1Weights = np.random.rand(self.size, self.size)
        self.o1Weights = np.random.rand(self.size, self.size)

        self.f2Weights = np.random.rand(self.size, self.prev.size)
        self.i2Weights = np.random.rand(self.size, self.prev.size)
        self.c2Weights = np.random.rand(self.size, self.prev.size)
        self.o2Weights = np.random.rand(self.size, self.prev.size)


class AI:
    def __init__(self, layers, optimizer="Adam", learningRate=0.0006):
        self.layers = layers
        self.optimizer = optimizer
        self.learningRate = learningRate

        if self.optimizer == "Adam":
            self.B1 = 0.9
            self.B2 = 0.999
            self.epsilon = 10**-8
            self.optimizerFunc = self._adam

            for layer in self.layers:
                layer.optAttrs["Mt"] = {}
                layer.optAttrs["Vt"] = {}

        elif self.optimizer == "Momentum":
            self.optimizerFunc = self._momentum
            for layer in self.layers:
                layer.optAttrs["vt"] = {}

        elif self.optimizer == "None":
            self.optimizerFunc = self._none
        else:
            raise ValueError(f"[ERROR] Invalid optimizer passed. You passed '{self.optimizer}'"
                             f", while only 'Adam', 'Momentum' and 'None' are allowed.")

        self._setupLayers()

    def _adam(self, layer, index, gradient):
        if index not in layer.optAttrs["Mt"]:
            layer.optAttrs["Mt"][index] = np.zeros(gradient.shape)
            layer.optAttrs["Vt"][index] = np.zeros(gradient.shape)

        layer.optAttrs["Mt"][index] = self.B1*layer.optAttrs["Mt"][index] + (1 - self.B1) * gradient
        layer.optAttrs["Vt"][index] = self.B2*layer.optAttrs["Vt"][index] + (1 - self.B2) * gradient**2

        Mht = layer.optAttrs["Mt"][index] / (1 - self.B1)
        Vht = layer.optAttrs["Vt"][index] / (1 - self.B2)

        Ms = (self.learningRate / (np.sqrt(Vht) + self.epsilon)) * Mht
        return Ms

    def _momentum(self, layer, index, gradient):
        if index not in layer.optAttrs["vt"]:
            layer.optAttrs["vt"][index] = np.zeros(gradient.shape)

        layer.optAttrs["vt"][index] = 0.9*layer.optAttrs["vt"][index] + self.learningRate*gradient
        return layer.optAttrs["vt"][index]

    def _none(self, layer, index, gradient):
        return self.learningRate * gradient

    def _setupLayers(self):
        if len(self.layers) < 3:
            raise ValueError(f"[ERROR] At least 3 layers are required")

        if self.layers[0].layerType != "InputLayer":
            raise ValueError(f"[ERROR] First layer isn't InputLayer")

        self.layers[0].setup_(None, self.layers[1], self.optimizerFunc)
        for i in range(1, len(self.layers) - 1):
            self.layers[i].setup_(self.layers[i - 1], self.layers[i + 1], self.optimizerFunc)
        self.layers[-1].setup_(self.layers[-2], None, self.optimizerFunc)

    def feedForward(self, inputState):
        if inputState.shape != self.layers[0].shape:
            raise ValueError(
                f"[ERROR] Feed-forward input's shape is not {self.layers[0].output.shape} but {inputState.shape}")

        self.layers[0].output = inputState.flatten()
        for i in range(1, len(self.layers)):
            self.layers[i].feedForward()
            #print(i, self.layers[i].weights, self.layers[i].biases)

    def train(self, dataset, epochs=1):
        # For each epoch
        for epoch in range(epochs):
            # print(f"Epoch {epoch + 1}/{epochs}")

            loss = 0

            # For each sentence
            for sentence in dataset:

                for layer in self.layers:
                    layer.reset()

                # For each word / timestep
                for inputState, actual in sentence:
                    self.feedForward(inputState)

                    loss += np.mean((actual - self.layers[-1].output) ** 2)

                    # lossDerivative = (2 / len(self.layers)) * (self.layers[-1].output - actual)
                    lossDerivative = self.layers[-1].output - actual
                    # errorL = lossDerivative * self.layers[-1].dActivation(self.layers[-1].zNeurons)
                    errorL = lossDerivative * self.layers[-1].dOutput

                    self.layers[-1].gradient = errorL

                    # L-1 .. 0
                    for i in range(len(self.layers) - 2, 0, -1):
                        layer = self.layers[i]
                        layer.calculateGradient()

                    # self.layers[-2].weights -= 0.001 * errorL * self.layers[-2].neurons

                # Sum gradients for each layer
                # gradients = np.sum(gradients, axis=0)

            for layer in self.layers:
                layer.updateParameters(sum([len(d) for d in dataset]))

            if loss < 0.0000001:
                print(f"Done at epoch {epoch+1}/{epochs} with loss {loss:.10f}")
                return

            if epoch % 100 == 0:
                # loss = loss / sum([len(d) for d in dataset])
                print(f"{epoch+1}/{epochs} {loss:.10f}")


ai = AI(layers=[
            InputLayer((3,)),
            # LSTMLayer(5),
            FFLayer(5, activation="Sigmoid"),
            FFLayer(2, activation="ReLU")
        ],
        optimizer="Adam",
        learningRate=0.001)

dataset = [
    [
        [np.array([0, 0, 1]), np.array([0, 0.5])],
        # [np.array([4, 1, 2]), np.array([1, 0])]
    ],
    # [
    #     [np.array([5, 2, 1]), np.array([1, 0.0])],
    #     [np.array([0, 1, 2]), np.array([1, 0.0])]
    # ]
]

ai.train(dataset, epochs=10000)
