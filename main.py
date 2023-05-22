
import numpy as np
import pickle
from nltk.tokenize import RegexpTokenizer


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


def dSoftmax(x):
    e = np.exp(x - np.max(x))
    return (e / np.sum(e)) * (1 - e / np.sum(e))


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dSigmoid(x):
    s = Sigmoid(x)
    return s * (1 - s)


def dTanH(x):
    return 1 - np.tanh(x) ** 2


def MSE(actual, pred):
    return (actual - pred)**2


def dMSE(actual, pred):
    return pred - actual


def CategoricalCrossEntropy(actual, pred):
    return -actual * np.log(pred)


def dCategoricalCrossEntropy(actual, pred):
    return -actual / (pred + 10**-8)


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
            self.dActivation = dSoftmax
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
    def __init__(self, layers, loss="MSE", optimizer="Adam", learningRate=0.0006):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.learningRate = learningRate

        if self.loss == "MSE":
            self.lossFunction = MSE
            self.dLossFunction = dMSE

        elif self.loss == "CategoricalCrossEntropy":
            self.lossFunction = CategoricalCrossEntropy
            self.dLossFunction = dCategoricalCrossEntropy

        else:
            raise ValueError(f"[ERROR] Invalid loss function passed. You passed '{self.loss}'"
                             f", while only 'MSE' and 'CategoricalCrossEntropy' are allowed.")

        if self.optimizer == "Adam":
            self.B1 = 0.9
            self.B2 = 0.999
            self.epsilon = 10**-8
            self.optimizerFunc = self._adam

            for layer in self.layers:
                layer.optAttrs["Mt"] = {}
                layer.optAttrs["Vt"] = {}

        elif self.optimizer == "RMSprop":
            self.epsilon = 10 ** -8
            self.gamma = 0.9
            self.optimizerFunc = self._rmsprop

            for layer in self.layers:
                layer.optAttrs["Eg2t"] = {}

        elif self.optimizer == "Momentum":
            self.optimizerFunc = self._momentum
            self.gamma = 0.9
            for layer in self.layers:
                layer.optAttrs["vt"] = {}

        elif self.optimizer == "None":
            self.optimizerFunc = self._none

        else:
            raise ValueError(f"[ERROR] Invalid optimizer passed. You passed '{self.optimizer}'"
                             f", while only 'Adam', 'RMSprop', 'Momentum' and 'None' are allowed.")

        self._setupLayers()

    def save(self, fn):
        with open(fn, "w") as f:
            f.write(pickle.dumps(self.__dict__))

    @classmethod
    def load(cls, fn):
        ai = cls.__new__(cls)
        with open(fn, "rb") as f:
            ai.__dict__ = pickle.load(f.read())

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

    def _rmsprop(self, layer, index, gradient):
        if index not in layer.optAttrs["Eg2t"]:
            layer.optAttrs["Eg2t"][index] = np.zeros(gradient.shape)

        layer.optAttrs["Eg2t"][index] = self.gamma*layer.optAttrs["Eg2t"][index] + 0.1*gradient**2
        return gradient * self.learningRate / np.sqrt(layer.optAttrs["Eg2t"][index] + self.epsilon)

    def _momentum(self, layer, index, gradient):
        if index not in layer.optAttrs["vt"]:
            layer.optAttrs["vt"][index] = np.zeros(gradient.shape)

        layer.optAttrs["vt"][index] = self.gamma*layer.optAttrs["vt"][index] + self.learningRate*gradient
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
                f"[ERROR] Feed-forward input's shape is {inputState.shape}, but got shape {self.layers[0].output.shape}"
                " in dataset.")

        self.layers[0].output = inputState.flatten()
        for i in range(1, len(self.layers)):
            self.layers[i].feedForward()
            #print(i, self.layers[i].weights, self.layers[i].biases)

    def train(self, dataset, epochs=1, mbSize=1, shuffle=False):
        if mbSize > len(dataset):
            raise ValueError(f"[ERROR] Mini-batch size ({mbSize}) is larger than the dataset's size ({len(dataset)})!")

        batchCount = int(np.ceil(len(dataset) / mbSize))

        print(f"""Training AI with parameters:
 - {epochs} epoch(s)
 - {batchCount} batch(es)
 - {mbSize} sample(s) per batch""")

        if len(dataset) % mbSize != 0:
            print(f" - {len(dataset) % mbSize} sample(s) for last batch")

        print(f""" - {self.learningRate} learning rate
 - '{self.optimizer}' optimization technique""")

        # For each epoch
        for epoch in range(epochs):
            # print(f"Epoch {epoch + 1}/{epochs}")

            loss = 0

            if shuffle:
                numpy.random.shuffle(dataset)

            for batch in range(batchCount):

                samples = dataset[batch*mbSize:min((batch+1)*mbSize, len(dataset))]
                batchSize = len(samples)

                # For each sentence
                for sentence in samples:

                    for layer in self.layers:
                        layer.reset()

                    # For each word / timestep
                    for inputState, actual in sentence:
                        self.feedForward(inputState)

                        # loss += np.mean((actual - self.layers[-1].output) ** 2)
                        loss += np.mean(self.lossFunction(actual, self.layers[-1].output))

                        # lossDerivative = (2 / len(self.layers)) * (self.layers[-1].output - actual)
                        lossDerivative = self.dLossFunction(actual, self.layers[-1].output)
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
                    layer.updateParameters(batchSize)

                if loss < 0.0000001:
                    print(f"Done at epoch {epoch+1}/{epochs} with loss {loss:.10f}")
                    return

            if epoch % 10 == 0:
                # loss = loss / sum([len(d) for d in dataset])
                print(f"{epoch+1}/{epochs} {loss:.10f}")


WORD_LENGTH = 5
SENTENCE_DEPTH = 5

with open("data.txt", "r") as f:
    text = f.read().lower()

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)
unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))
# print(words)
# print(unique_words)

dataset = []

# for i in range(len(words) - SENTENCE_DEPTH):
#     dataset.append()

next_word = []
prev_words = []
for j in range(len(words) - SENTENCE_DEPTH):
    prev_words.append(words[j:j + SENTENCE_DEPTH])
    next_word.append(words[j + SENTENCE_DEPTH])
print(prev_words[1])
print(next_word[1])


exit()


ai = AI(layers=[
            InputLayer((WORD_LENGTH,)),
            LSTMLayer(WORD_LENGTH),
            # FFLayer(6, activation="Sigmoid"),
            FFLayer(2, activation="Softmax")
        ],
        loss="CategoricalCrossEntropy",
        optimizer="RMSprop",
        learningRate=0.01)

# dataset = [
#     [
#         [np.array([0, 0, 1]), np.array([0, 1])],
#         #[np.array([4, 1, 2]), np.array([1, 0])]
#     ],
#     # [
#     #     [np.array([5, 2, 1]), np.array([1, 0.0])],
#     #     [np.array([0, 1, 2]), np.array([1, 0.0])]
#     # ],
# ]

ai.train(dataset, epochs=2, mbSize=128, shuffle=True)

ai.save("shatgpt.model")
