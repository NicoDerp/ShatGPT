
import numpy as np
import pickle
import matplotlib.pyplot as plt


def LinearActivation(x):
    return x


def dLinearActivation(x):
    return 1


def ReLU(x):
    return np.maximum(x, 0)


def dReLU(x):
    return np.greater(x, 0).astype(int)


def Softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def dSoftmax(x):
    s = np.exp(x - np.max(x)) / np.sum(x)
    return s * (1 - s)


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dSigmoid(x):
    s = Sigmoid(x)
    return s * (1 - s)


def dTanH(x):
    return 1 - np.tanh(x) ** 2


def MSE(actual, pred):
    return np.mean((actual - pred)**2)


def dMSE(actual, pred):
    return pred - actual


def CategoricalCrossEntropy(actual, pred):
    # pred = np.clip(pred, 1e-12, 1.0 - 1e-12)
    return -np.sum(actual * np.log(pred + 10**-100))
    # return -actual * np.log(pred)


def dCategoricalCrossEntropy(actual, pred):
    # pred = np.clip(pred, 1e-12, 1.0 - 1e-12)
    # return -actual / (pred + 10**-100)
    return actual / (pred + 10**-100)
    # return -actual / pred


class Layer:
    def __init__(self, layerType, size, activation):
        self.layerType = layerType
        self.size = size
        self.output = np.zeros(size)
        self.dOutput = np.zeros(size)
        self.gradient = np.zeros(size)
        self.sGradient = np.zeros(size)
        self.prev = None
        self.next = None
        self.optimizerFunc = None
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
        self.output = np.zeros(self.size)
        self.dOutput = np.zeros(self.size)

        self.gradient = np.zeros(self.size)
        self.sGradient = np.zeros(self.size)

    def setup_(self, prev, nextL):
        self.prev = prev
        self.next = nextL

        self.reset()

    def setupOptimizer_(self, optimizerFunc):
        self.optimizerFunc = optimizerFunc


class InputLayer(Layer):
    def __init__(self, shape):
        super().__init__("InputLayer", np.prod(shape), "Linear")
        self.shape = shape


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

    def calculateGradient(self):
        self.gradient = self.next.weights.T.dot(self.next.gradient) * self.dOutput
        self.sGradient += self.gradient

    def updateParameters(self, n):
        # Average the gradients
        self.sGradient /= n
        # self.weights -= self.optimizerFunc(self, 0, self.gradient.dot(self.output[np.newaxis].T))
        self.weights -= self.optimizerFunc(self, 0, self.sGradient.dot(self.output.reshape((-1, 1))))
        self.biases -= self.optimizerFunc(self, 1, self.sGradient)

    def setup_(self, prev, nextL):
        super().setup_(prev, nextL)

        self.weights = np.random.rand(self.size, self.prev.size)


class LSTMLayer(Layer):
    def __init__(self, size, activation="Linear"):
        super().__init__("FFLayer", size, activation)

        self.lOutput = np.zeros(size)
        self.lStates = np.zeros(size)

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

        # self.f1WeightsGr = None
        # self.i1WeightsGr = None
        # self.c1WeightsGr = None
        # self.o1WeightsGr = None
        #
        # self.f2WeightsGr = None
        # self.i2WeightsGr = None
        # self.c2WeightsGr = None
        # self.o2WeightsGr = None
        #
        # self.fBiasesGr = np.zeros(self.size)
        # self.iBiasesGr = np.zeros(self.size)
        # self.cBiasesGr = np.zeros(self.size)
        # self.oBiasesGr = np.zeros(self.size)

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

        self.lOutput = self.output
        self.lStates = self.states
        self.output = self.ft * self.output + self.it * self.gt
        self.dOutput = self.dActivation(self.output)
        self.output = self.activation(self.output)
        self.states = self.ot * np.tanh(self.output)

    def calculateGradient(self):
        # nGradient = self.next.weights.T.dot(self.next.gradient)
        self.gradient = self.next.weights.T.dot(self.next.gradient) * self.dOutput
        self.sGradient += self.gradient

        # self.f1WeightsGr += np.dot(gf, states.T)
        # self.f2WeightsGr += np.dot(gf, prevOutput.T)
        # self.fBiasesGr += gf
        #
        # self.i1WeightsGr += np.dot(gi, states.T)
        # self.i2WeightsGr += np.dot(gi, prevOutput.T)
        # self.iBiasesGr += gi
        #
        # self.c1WeightsGr += np.dot(gc, states.T)
        # self.c2WeightsGr += np.dot(gc, prevOutput.T)
        # self.cBiasesGr += gc
        #
        # self.o1WeightsGr += np.dot(go, states.T)
        # self.o2WeightsGr += np.dot(go, prevOutput.T)
        # self.oBiasesGr += go

    def updateParameters(self, n):
        # Output
        go = self.sGradient * np.tanh(self.output) * dSigmoid(self.zo)

        # Forget
        gf = self.sGradient * self.ot * dTanH(self.output) * self.lOutput * dSigmoid(self.zf)

        # Input
        gi = self.sGradient * self.ot * dTanH(self.output) * self.gt * dSigmoid(self.zi)

        # C
        gc = self.sGradient * self.ot * dTanH(self.output) * self.it * dTanH(self.zg)

        prevOutput = self.prev.output.reshape((1, -1))
        states = self.states.reshape((1, -1))

        self.f1Weights -= self.optimizerFunc(self, 0, np.dot(gf, states.T))
        self.f2Weights -= self.optimizerFunc(self, 1, np.dot(gf, prevOutput.T))
        self.fBiases -= self.optimizerFunc(self, 2, gf)

        self.i1Weights -= self.optimizerFunc(self, 3, np.dot(gi, states.T))
        self.i2Weights -= self.optimizerFunc(self, 4, np.dot(gi, prevOutput.T))
        self.iBiases -= self.optimizerFunc(self, 5, gi)

        self.c1Weights -= self.optimizerFunc(self, 6, np.dot(gc, states.T))
        self.c2Weights -= self.optimizerFunc(self, 7, np.dot(gc, prevOutput.T))
        self.cBiases -= self.optimizerFunc(self, 8, gc)

        self.o1Weights -= self.optimizerFunc(self, 9, np.dot(go, states.T))
        self.o2Weights -= self.optimizerFunc(self, 10, np.dot(go, prevOutput.T))
        self.oBiases -= self.optimizerFunc(self, 11, go)

        # self.f1Weights -= self.optimizerFunc(self, 0, self.f1WeightsGr)
        # self.f2Weights -= self.optimizerFunc(self, 1, self.f2WeightsGr)
        # self.fBiases -= self.optimizerFunc(self, 2, self.fBiasesGr)
        #
        # self.i1Weights -= self.optimizerFunc(self, 3, self.i1WeightsGr)
        # self.i2Weights -= self.optimizerFunc(self, 4, self.i2WeightsGr)
        # self.iBiases -= self.optimizerFunc(self, 5, self.iBiasesGr)
        #
        # self.c1Weights -= self.optimizerFunc(self, 6, self.c1WeightsGr)
        # self.c2Weights -= self.optimizerFunc(self, 7, self.c2WeightsGr)
        # self.cBiases -= self.optimizerFunc(self, 8, self.cBiasesGr)
        #
        # self.o1Weights -= self.optimizerFunc(self, 9, self.o1WeightsGr)
        # self.o2Weights -= self.optimizerFunc(self, 10, self.o2WeightsGr)
        # self.oBiases -= self.optimizerFunc(self, 11, self.oBiasesGr)

    def reset(self):
        super().reset()

        self.states = np.zeros(self.size)

        self.lOutput = np.zeros(self.size)
        self.lStates = np.zeros(self.size)

        # self.f1WeightsGr = np.zeros((self.size, self.size))
        # self.f2WeightsGr = np.zeros((self.size, self.prev.size))
        # self.fBiasesGr = np.zeros(self.size)
        #
        # self.i1WeightsGr = np.zeros((self.size, self.size))
        # self.i2WeightsGr = np.zeros((self.size, self.prev.size))
        # self.iBiasesGr = np.zeros(self.size)
        #
        # self.c1WeightsGr = np.zeros((self.size, self.size))
        # self.c2WeightsGr = np.zeros((self.size, self.prev.size))
        # self.cBiasesGr = np.zeros(self.size)
        #
        # self.o1WeightsGr = np.zeros((self.size, self.size))
        # self.o2WeightsGr = np.zeros((self.size, self.prev.size))
        # self.oBiasesGr = np.zeros(self.size)

    def setup_(self, prev, nextL):
        super().setup_(prev, nextL)

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

        self._setupLoss()
        self._setupOptimizer()
        self._setupLayers()

    def save(self, fn):
        with open(fn, "wb") as f:
            f.write(pickle.dumps(self.__dict__))

    @classmethod
    def load(cls, fn):
        ai = cls.__new__(cls)
        with open(fn, "rb") as f:
            ai.__dict__ = pickle.load(f)

        ai._setupLoss()
        ai._setupOptimizer()

        for layer in ai.layers:
            layer.setupOptimizer_(ai.optimizerFunc)
        # ai._setupLayers()

        return ai

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

        self.layers[0].setup_(None, self.layers[1])
        for i in range(1, len(self.layers) - 1):
            self.layers[i].setup_(self.layers[i - 1], self.layers[i + 1])
        self.layers[-1].setup_(self.layers[-2], None)

        for layer in self.layers:
            layer.setupOptimizer_(self.optimizerFunc)

    def _setupOptimizer(self):
        if self.optimizer == "Adam":
            self.B1 = 0.9
            self.B2 = 0.999
            self.epsilon = 10**-8
            self.optimizerFunc = self._adam

            for layer in self.layers:
                layer.optAttrs["Mt"] = {}
                layer.optAttrs["Vt"] = {}

        elif self.optimizer == "RMSprop":
            self.epsilon = 10**-8
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

    def _setupLoss(self):
        if self.loss == "MSE":
            self.lossFunction = MSE
            self.dLossFunction = dMSE

        elif self.loss == "CategoricalCrossEntropy":
            self.lossFunction = CategoricalCrossEntropy
            self.dLossFunction = dCategoricalCrossEntropy

        else:
            raise ValueError(f"[ERROR] Invalid loss function passed. You passed '{self.loss}'"
                             f", while only 'MSE' and 'CategoricalCrossEntropy' are allowed.")

    def feedForward(self, inputState):
        if inputState.shape != self.layers[0].shape:
            raise ValueError(
                f"[ERROR] Feed-forward input's shape is {inputState.shape}, but got shape {self.layers[0].output.shape}"
                " in dataset.")

        self.layers[0].output = inputState.flatten()
        for i in range(1, len(self.layers)):
            self.layers[i].feedForward()
            # print(i, self.layers[i].weights, self.layers[i].biases)

    def clean(self):
        for layer in self.layers:
            layer.reset()

    def predictNextWord(self, inputs, wordList, n=1):
        self.clean()
        for w in inputs:
            self.feedForward(w)

        prediction = self.layers[-1].output
        predIndexes = np.argpartition(prediction, -n)[-n:]
        sortedPredIndexes = reversed(predIndexes[np.argsort(prediction[predIndexes])])
        predWords = [wordList[i] for i in sortedPredIndexes]
        return predWords

    def train(self, data, labels, epochs=1, mbSize=1, shuffle=False):
        # if data.shape != labels.shape:
        #    raise ValueError(f"[ERROR] Data shape {data.shape} does not match label shape {labels.shape}")

        if mbSize <= 0:
            raise ValueError(f"[ERROR] Mini-batch size must be larger than 0. Currently it is {mbSize}")

        datasetSize = data.size
        if mbSize > datasetSize:
            raise ValueError(f"[ERROR] Mini-batch size ({mbSize}) is larger than the dataset's size ({datasetSize})!")

        batchCount = int(np.ceil(datasetSize / mbSize))

        print(f"""Training AI with parameters:
 - {epochs} epoch(s)
 - {batchCount} batch(es)
 - {mbSize} sample(s) per batch""")

        if datasetSize % mbSize != 0:
            print(f" - {datasetSize % mbSize} sample(s) for last batch")

        print(f""" - {self.learningRate} learning rate
 - '{self.optimizer}' optimization technique
 - Dataset with {datasetSize} sentences""")

        losses = []
        accuracies = []

        # For each epoch
        for epoch in range(epochs):
            # print(f"Epoch {epoch + 1}/{epochs}")

            loss = 0
            accuracy = 0

            if shuffle:
                indices = np.arange(data.shape[0])
                np.random.shuffle(indices)
                data = data[indices]
                labels = labels[indices]

            for batch in range(batchCount):
                acc = 0

                samples = data[batch*mbSize:min((batch+1)*mbSize, datasetSize)]
                label = labels[batch]
                batchSize = len(samples)

                # For each sentence
                for sentence in samples:

                    self.clean()

                    # For each word / timestep predict
                    for inputState in sentence:
                        self.feedForward(inputState)

                    # loss += np.mean((actual - self.layers[-1].output) ** 2)
                    loss += self.lossFunction(label, self.layers[-1].output)

                    predictedIndex = np.argmax(self.layers[-1].output)
                    actualIndex = np.argmax(label)
                    # print(predictedIndex)
                    # print(actual, actualIndex)
                    # print(self.layers[-1].output)
                    if predictedIndex == actualIndex:
                        # print("YUUUUUH")
                        acc += 1

                    # lossDerivative = (2 / len(self.layers)) * (self.layers[-1].output - actual)
                    lossDerivative = self.dLossFunction(label, self.layers[-1].output)
                    # errorL = lossDerivative * self.layers[-1].dActivation(self.layers[-1].zNeurons)
                    errorL = lossDerivative * self.layers[-1].dOutput

                    self.layers[-1].gradient = errorL
                    self.layers[-1].sGradient += errorL

                    # L-1 .. 0
                    for i in range(len(self.layers) - 2, 0, -1):
                        layer = self.layers[i]
                        layer.calculateGradient()

                    # self.layers[-2].weights -= 0.001 * errorL * self.layers[-2].neurons

                for layer in self.layers:
                    layer.updateParameters(batchSize)

                if loss < 0.0000001:
                    print(f"Done at epoch {epoch+1}/{epochs} with loss {loss:.10f}")
                    break

                # if batch % 4 == 0:
                #     # loss = loss / sum([len(d) for d in dataset])
                #     print(f"Batch {batch + 1}/{batchCount} {loss:.10f}")

                accuracy += acc / batchSize

            if epoch % 10 == 0:
                # loss = loss / sum([len(d) for d in dataset])
                print(f"Epoch {epoch+1}/{epochs} {loss:.10f}")

            losses.append(loss / batchCount)
            accuracies.append(accuracy / batchCount)

        print("\nTraining complete!")

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax[0].plot(np.arange(0, epochs), losses, label="Loss")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")

        ax[1].plot(np.arange(0, epochs), accuracies, label="Accuracy")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("accuracy")
        # plt.title("ShatGPT stats")
        fig.tight_layout(pad=10.0)
        plt.legend()
        plt.grid()
        plt.show()
