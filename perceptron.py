import numpy as np
from constants import *


class Perceptron:
    '''
    by Geoffrey Qian
    The model class
    '''
    hidden_layer: np.ndarray
    hidden_layer2: np.ndarray
    bias: np.float32
    lr: np.float32
    weight_decay: np.float32

    def __init__(self, lr: np.float32, weight_decay: np.float32):
        '''Perceptron constructor'''
        self.lr = lr
        self.weight_decay = weight_decay
        self.bias = np.float32(0)
        self.hidden_layer = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    def forward(self, input: np.ndarray) -> bool:
        '''the inference step of the model'''
        return self.bias + np.multiply(input, self.hidden_layer).sum() > 0

    def backward(self, input: np.ndarray, correct_ans: bool, model_ans: bool):
        '''the learning step of the model'''
        self.hidden_layer = self.hidden_layer + self.lr * input * \
            ((1 if correct_ans else 0) - (1 if model_ans else 0))
        self.bias = self.bias + self.lr * \
            ((1 if correct_ans else 0) - (1 if model_ans else 0))
        self.hidden_layer = self.hidden_layer * (1-self.weight_decay)
        self.bias = self.bias * (1-self.weight_decay)
