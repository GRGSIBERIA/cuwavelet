/*
    @brief ウェーブレット変換を行う
    @param[out] transform Wavelet変換した結果
    @param[in] wavelets スケーリングした後のWavelet
    @param[in] waveform 変換の対象となる波形
    @param[in] timeN 時刻の長さ
    @param[in] scaleN スケールの数
*/
__global__ void wavelet_transform(
        float* transform, float* wavelets, float* waveform,
        int timeN, int scaleN
    ) {
    int t = blockDim.x * blockIdx.y + threadIdx.x;    // 時間軸
    int s = blockDim.y * blockIdx.y + threadIdx.y;    // スケール軸
    int idx = scaleN * s + t;

    // 畳み込み積分, f(t)g(i-t)の形式
    float total = 0;
    for (int i = t; i < timeN; ++i) {
        total += waveforms[t] * wavelets[s * scaleN + i - t];
    }
    transform[idx] = total;
}

/*
有毛細胞の数は11,500個
記憶している長さ、44,100Hz(1秒)
計算量は480,000,000
グラボのCUDAコア1000個で割ると480,000
*/