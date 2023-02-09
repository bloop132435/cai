import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
def to_string(arr: np.ndarray)->str:
    '''convert a numpy array to a string representation. Used for debugging'''
    assert len(arr.shape)==2
    return np.array2string(arr,separator='')

#  def save_heatmap(arr: np.ndarray,filename:str):
    #  img_arr = np.zeros((arr.shape[0],arr.shape[1],3),dtype=np.uint8)
    #  img_arr[:,:,0] = 129 + 256*arr
    #  img = Image.fromarray(img_arr)
    #  img.save(filename)
    #  img.show()
    #  pass
def save_heatmap(arr: np.ndarray,filename:str):
    '''convert a numpy array to a PNG'''
    plt.clf()
    plt.imshow(arr, cmap='inferno')
    plt.colorbar()
    plt.savefig(filename)
    #  plt.show()
