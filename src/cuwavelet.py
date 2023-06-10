import pycuda.autoinit
import pycuda.driver as driver
import numpy as np
from typing import List
from scipy.signal import morlet
import matplotlib.pyplot as plt

class PyWavelet:
    def __init__(self, window_length: int, intervals: List) -> None:
        self.__waveletsRe = []
        self.__waveletsIm = []
        self.__window_length = window_length

        for interval in intervals:
            wavelet = morlet(window_length, 5, interval)
            re = [x.real for x in wavelet]
            im = [x.imag for x in wavelet]
            
            self.__waveletsRe.append(re)
            self.__waveletsIm.append(im)


if __name__ == "__main__":
    plt.figure()
    
    Tf = 1. / 44100
    for i in [4000, 8000, 12000]:
        
        length = 44100. / i
        print(length)
        x = morlet(44100, 5, i)
        print(x[0].real, x[0].imag)
        plt.plot(x, marker="x", label=str(i))

    # f = 2*s*w*r / M
    # f = 2*1*5*44100 / 44.1
    
    
    plt.legend()
    plt.tight_layout()
    plt.show()