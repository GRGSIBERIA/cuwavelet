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
        self.__waveform = np.zeros(window_length)

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
        self.__waveletsIm = np.copy(self.__waveletsRe)
        self.__resultRe = np.zeros_like(shape=(self.__wavelet_scales, self.__window_length), dtype=np.float32)
        self.__resultIm = np.copy(self.__resultRe)
        self.__resultAbs = np.copy(self.__resultRe)

        with open("./src/wavelet.cu", "rt") as f:
            s = f.read()
            self.__module = SourceModule(s)
        
        self.__func = self.__module.get_function("wavelet_transform")

        self.__gpuRe = cuda.In(self.__waveletsRe)
        self.__gpuIm = cuda.In(self.__waveletsIm)
        
        self.__gpu_window_length = cuda.In(self.__window_length)
        self.__gpu_scale_size = cuda.In(self.__wavelet_scales)

    
    def compute(self, compute_waveform):
        # 循環リストの処理
        target = None
        if len(compute_waveform) > self.__window_length:
            target = compute_waveform[len(compute_waveform) - self.__window_length:]
        else:
            if type(compute_waveform) == type(np.array([])):
                target = np.concatenate(self.__waveform, compute_waveform)
            elif type(compute_waveform) == type([]):
                target += compute_waveform
            target = target[len(compute_waveform):]
        
        if target == None:
            raise TypeError(f"compute_waveform is not list or ndarray: {type(compute_waveform)}")

        self.__waveform = np.copy(target)

        # resultにWavelet変換の結果を出力する
        self.__func(
            cuda.Out(self.__resultRe), cuda.Out(self.__resultIm),
            cuda.Out(self.__resultAbs),
            self.__gpuRe, self.__gpuIm,
            cuda.In(self.__waveform),
            self.__gpu_window_length, self.__gpu_scale_size,
            block=(self.__window_length, self.__wavelet_scales, 1), grid=(1,1))
        


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