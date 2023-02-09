from constants import *
import numpy as np
from typing import Tuple
from utils import to_string
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)


def square(center: Tuple[int, int], size: int) -> np.ndarray:
    '''function that generates a square image'''
    ans = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    low_x = center[0] - size
    low_y = center[1] - size
    hi__x = center[0] + size
    hi__y = center[1] + size
    assert low_x >= 0
    assert low_y >= 0
    assert hi__x < IMAGE_SIZE
    assert hi__y < IMAGE_SIZE
    ans[low_x:hi__x+1, low_y:hi__y+1] = 1
    return ans


def circle(center: Tuple[int, int], radius: int) -> np.ndarray:
    '''function that generates a circle image'''
    ans = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    indices = np.indices(ans.shape)
    distance = ((indices[0]-center[0])**2 +
                (indices[1]-center[1])**2) <= radius**2
    x = indices[0, distance]
    y = indices[1, distance]
    ans[x, y] = 1
    return ans


if __name__ == "__main__":
    print(to_string(np.where(circle((15, 15), 12) == 1, '#', ' ')))
#  print(square((5,5),6))
