import pycuda.autoinit
import pycuda.driver as driver
import numpy as np
from typing import List
from scipy.signal import morlet
import matplotlib.pyplot as plt

class PyWavelet:
    def __init__(self, window_length: int, sampling_rate: int, intervals: List) -> None:
        self.__waveletsRe = []
        self.__waveletsIm = []
        self.__wavelet_length = []
        self.__window_length = window_length
        self.__fs = sampling_rate

        for interval in intervals:
            length = self.__fs / interval

            wavelet = morlet(length * 4, 2, 1)
            self.__wavelet_length.append(len(wavelet))
            zeros = np.zeros(self.__window_length - len(re)).tolist()

            re = [x.real for x in wavelet] + zeros
            im = [x.imag for x in wavelet] + zeros

            self.__waveletsRe.append(re)
            self.__waveletsIm.append(im)

            


if __name__ == "__main__":
    plt.figure()
    
    Tf = 1. / 44100
    for i in [16000]:
        
        length = 44100. / i
        print(length)
        x = morlet(int(length) * 4, 2, 1)
        re = [re.real for re in x]
        im = [im.imag for im in x]
        plt.plot(re, marker="x", label=str(i))
        plt.plot(im, marker="x", label=str(i))

        
    # f = 2*s*w*sr / length
    # f = 2*1*5*44100 / 44.1
    # f = 
    
    
    plt.legend()
    plt.tight_layout()
    plt.show()