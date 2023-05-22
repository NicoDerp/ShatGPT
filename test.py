
import numpy as np
from aiLib import *


ai = AI.load("shatgpt.model")

print(ai.layers[2].weights)

