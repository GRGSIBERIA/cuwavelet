/*
    Parameters
        transform: float**
            Wavelet変換した結果
        wavelets: float**
            スケーリングした後のWavelet
        waveform: float*
            変換の対象となる波形
*/
__global__ void wavelet_transform(float** transform, float** wavelets, float* waveform) {
    // threadIdx.x, blockDim.x
    int t = threadIdx.x;    // 時間軸
    int s = threadIdx.y;    // スケール軸

    // 畳み込み積分
    float total = 0;
    for (int i = 0; i < length; ++i) {
        total += waveforms[t] * wavelets[s][i];
    }
    transform[s][t] = total;
}

/*
有毛細胞の数は11,500個
記憶している長さ、44,100Hz(1秒)
計算量は480,000,000
グラボのCUDAコア1000個で割ると480,000
*/