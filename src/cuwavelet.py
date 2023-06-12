import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from typing import List
from scipy.signal import morlet
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule

class PyWavelet:
    def __init__(self, window_length: int, sampling_rate: int, intervals: List) -> None:
        self.__waveletsRe = []
        self.__waveletsIm = []
        self.__wavelet_scales = len(intervals)
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
        
        self.__waveletsRe = np.array(self.__waveletsRe, dtype=np.float32)
        self.__waveletsIm = np.array(self.__waveletsIm, dtype=np.float32)
        self.__resultRe = np.zeros(self.__window_length, dtype=np.float32)
        self.__resultIm = np.zeros(self.__window_length, dtype=np.float32)

        with open("./src/wavelet.cu", "rt") as f:
            s = f.read()
            self.__module = SourceModule(s)
        
        self.__func = self.__module.get_function("wavelet_transform")

        self.__regpu = cuda.mem_alloc(self.__waveletsRe.nbytes)
        self.__imgpu = cuda.mem_alloc(self.__waveletsIm.nbytes)
        self.__resim = cuda.mem_alloc(self.__resultRe.nbytes)
        self.__resre = cuda.mem_alloc(self.__resultIm.nbytes)

        cuda.memcpy_htod(self.__regpu, self.__waveletsRe)
        cuda.memcpy_htod(self.__imgpu, self.__waveletsIm)
    
    def __del__(self):
        self.__regpu.free()
        self.__imgpu.free()
        self.__resim.free()
        self.__resre.free()
    
    def compute(self, compute_waveform):
        pass

            


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