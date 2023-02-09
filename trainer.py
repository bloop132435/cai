import numpy as np
from constants import *
from perceptron import Perceptron
from shape import square, circle
from utils import save_heatmap
import random


def train(model: Perceptron, iterations=100):
    '''train the model input on a number of iterations of images'''
    random.seed()
    for _ in range(iterations):
        #  pos_x = random.randint(1, IMAGE_SIZE-1)
        #  pos_y = random.randint(1, IMAGE_SIZE-1)
        pos_x = IMAGE_SIZE//2
        pos_y = IMAGE_SIZE//2
        largest_possible = min(
            min(pos_x, IMAGE_SIZE-pos_x-1),
            min(pos_y, IMAGE_SIZE-pos_y-1)
        )
        size = random.randint(0, largest_possible)
        square_pic = square((pos_x, pos_y), size)
        circle_pic = circle((pos_x, pos_y), size)
        s_ans = model.forward(square_pic)
        model.backward(square_pic, True, s_ans)
        c_ans = model.forward(circle_pic)
        model.backward(circle_pic, False, c_ans)


def test(model: Perceptron, iterations=100) -> float:
    '''test the model input on a number of iterations of images'''
    cnt = 0
    random.seed()
    for _ in range(iterations):
        #  pos_x = random.randint(1, IMAGE_SIZE-1)
        #  pos_y = random.randint(1, IMAGE_SIZE-1)
        pos_x = IMAGE_SIZE//2
        pos_y = IMAGE_SIZE//2
        largest_possible = min(
            min(pos_x, IMAGE_SIZE-pos_x-1),
            min(pos_y, IMAGE_SIZE-pos_y-1)
        )
        size = random.randint(0, largest_possible)
        square_pic = square((pos_x, pos_y), size)
        circle_pic = circle((pos_x, pos_y), size)
        s_ans = model.forward(square_pic)
        cnt += s_ans
        c_ans = model.forward(circle_pic)
        cnt += not c_ans
    return cnt/(iterations*2)

if __name__ == "__main__":
    avg = 0
    cnt = 0
    total = 0
    p = Perceptron(np.float32(0.01), weight_decay=np.float32(0.0005))
    for _ in range(100):
        train(p, 100)
        f = test(p, 300)
        print(f*100)

    #  with open('1000x1000_image_weights.npy', 'wb') as fl:
        #  np.save(fl, p.hidden_layer)
    #  with open('1000x1000_image_bias.npy', 'w') as fl:
        #  fl.write(str(float(p.bias)))

    #  train(p,1)

    save_heatmap(p.hidden_layer,"weight.png")
