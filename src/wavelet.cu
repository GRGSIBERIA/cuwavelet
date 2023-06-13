/*
    @brief ウェーブレット変換を行う
    @param[out] transform Wavelet変換した結果
    @param[in] waveletsRe スケーリングした後のWaveletの実部
    @param[in] waveletsIm スケーリングした後のWaveletの虚部
    @param[in] waveform 変換の対象となる波形
    @param[in] timeN 時刻の長さ
    @param[in] scaleN スケールの数
*/
__global__ void wavelet_transform(
        float* transformRe, float* transformIm, float* transformAbs,
        float* waveletsRe, float* waveletsIm,
        float* waveform,
        int timeN, int scaleN
    ) {
    int t = blockDim.x * blockIdx.y + threadIdx.x;    // 時間軸
    int s = blockDim.y * blockIdx.y + threadIdx.y;    // スケール軸
    int idx = scaleN * s + t;

    // 畳み込み積分, f(t)g(i-t)の形式
    float totalRe = 0;
    float totalIm = 0;
    int si = s * scaleN;
    for (int i = t; i < timeN; ++i) {
        totalRe += waveforms[t] * waveletsRe[si + i - t];
        totalIm += waveforms[t] * waveletsIm[si + i - t];
    }
    
    transformRe[idx] = totalRe;
    transformIm[idx] = totalIm;
    transformAbs[idx] = sqrt(totalRe*totalRe + totalIm*totalIm);
}

/*
複素数の積は以下で定義される
z1 * z2 = (x1*x2-y1*y2)+(x1*y2+x2*y1)i
このとき、z2の虚部がなく、y2 = 0のとき
z1 * z2 = (x1*x2)+(x2*y1)i
z1 * z2 = (R1*R2)+(R2*I1)i
*/

/*
有毛細胞の数は11,500個
記憶している長さ、44,100Hz(1秒)
計算量は480,000,000
グラボのCUDAコア1000個で割ると480,000
*/