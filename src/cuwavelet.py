import pycuda.autoinit
import pycuda.driver as driver
import numpy as np
from typing import List
from scipy.signal import morlet
import matplotlib.pyplot as plt

class PyWavelet:
    def __init__(self, sampling_freqency: int, intervals: List) -> None:
        self.__fs = sampling_freqency
        self.__waveletsRe = []
        self.__waveletsIm = []

        for interval in intervals:
            length = sampling_freqency / interval
            length = np.round(length)

            wavelet = morlet(length, 5, 1)
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
        x = morlet(int(length), 5, 1)
        print(x[0].real, x[0].imag)
        plt.plot(x, marker="x", label=str(i))

    # f = 2*s*w*r / M
    # f = 2*1*5*44100 / 44.1
    
    
    plt.legend()
    plt.tight_layout()
    plt.show()