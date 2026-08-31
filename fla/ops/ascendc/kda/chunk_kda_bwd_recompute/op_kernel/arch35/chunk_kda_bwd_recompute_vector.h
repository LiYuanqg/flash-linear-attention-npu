/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH35_VECTOR_H

#include "../chunk_kda_bwd_recompute_struct.h"
#include "../chunk_kda_bwd_recompute_common.h"
#include "chunk_kda_bwd_recompute_common.h"
#include "chunk_kda_bwd_recompute_regbase.h"
#include "catlass/arch/cross_core_sync.hpp"

using namespace AscendC;

namespace KDA {

template <typename QkType, typename GateType, typename BetaType>
class ChunkKdaBwdRecomputeVectorProcess {
public:
    __aicore__ inline ChunkKdaBwdRecomputeVectorProcess(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR aLog, GM_ADDR dtBias,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR gk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg,
        GM_ADDR workspace, const ChunkKdaBwdRecomputeTilingData *tiling)
        : q_(q), k_(k), v_(v), g_(g), beta_(beta), aLog_(aLog), dtBias_(dtBias),
          cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices), gk_(gk), w_(w), u_(u), qg_(qg), kg_(kg),
          workspace_(workspace), tiling_(tiling)
    {
        (void)workspace_;
        (void)w_;
        (void)u_;
    }

    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe_ = pipe;
        qTensor_.SetGlobalBuffer((__gm__ QkType *)q_);
        kTensor_.SetGlobalBuffer((__gm__ QkType *)k_);
        vTensor_.SetGlobalBuffer((__gm__ QkType *)v_);
        gTensor_.SetGlobalBuffer((__gm__ GateType *)g_);
        betaTensor_.SetGlobalBuffer((__gm__ BetaType *)beta_);
        if (aLog_ != nullptr) {
            aLogTensor_.SetGlobalBuffer((__gm__ float *)aLog_);
        }
        if (dtBias_ != nullptr) {
            dtBiasTensor_.SetGlobalBuffer((__gm__ float *)dtBias_);
        }
        gkTensor_.SetGlobalBuffer((__gm__ float *)gk_);
        qgTensor_.SetGlobalBuffer((__gm__ QkType *)qg_);
        kgTensor_.SetGlobalBuffer((__gm__ QkType *)kg_);

        B_ = static_cast<uint64_t>(tiling_->B);
        Hk_ = static_cast<uint64_t>(tiling_->Hk);
        Hv_ = static_cast<uint64_t>(tiling_->Hv);
        hvPerHk_ = static_cast<uint64_t>(tiling_->hvPerHk);
        T_ = static_cast<uint64_t>(tiling_->T);
        chunkNum_ = static_cast<uint64_t>(tiling_->chunkNum);
        chunkSize_ = static_cast<uint64_t>(tiling_->chunkSize);
        useGate_ = tiling_->useGateInKernel != 0;
        hasDtBias_ = tiling_->hasDtBias != 0;
        {
            union {
                uint32_t u;
                float f;
            } conv;
            conv.u = static_cast<uint32_t>(tiling_->lowerBoundBits);
            lowerBound_ = conv.f;
        }
        expA_ = 1.0f;

        l1Buffer_ = LocalTensor<uint8_t>(TPosition::A1, 0, 512 * 1024);
        chunkReadyFlag_ = Catlass::Arch::CrossCoreFlag(KdaBwdRecomputeArch35::kChunkReadyFlag);
        chunkFreeFlag_ = Catlass::Arch::CrossCoreFlag(KdaBwdRecomputeArch35::kChunkFreeFlag);
    }

    __aicore__ inline void Process()
    {
        pipe_->InitBuffer(betaRawBuf_, KdaBwdRecomputeArch35::kBt * sizeof(BetaType));
        pipe_->InitBuffer(gateBuf_, KdaBwdRecomputeArch35::kBt * KdaBwdRecomputeArch35::kK * sizeof(float));
        pipe_->InitBuffer(qBuf_, KdaBwdRecomputeArch35::kBt * KdaBwdRecomputeArch35::kK * sizeof(QkType));
        pipe_->InitBuffer(kBuf_, KdaBwdRecomputeArch35::kBt * KdaBwdRecomputeArch35::kK * sizeof(QkType));
        pipe_->InitBuffer(vBuf_, KdaBwdRecomputeArch35::kBt * KdaBwdRecomputeArch35::kV * sizeof(QkType));
        pipe_->InitBuffer(qgBuf_, KdaBwdRecomputeArch35::kBt * KdaBwdRecomputeArch35::kK * sizeof(QkType));
        pipe_->InitBuffer(kbgBuf_, KdaBwdRecomputeArch35::kBt * KdaBwdRecomputeArch35::kK * sizeof(QkType));
        pipe_->InitBuffer(kgBuf_, KdaBwdRecomputeArch35::kBt * KdaBwdRecomputeArch35::kK * sizeof(QkType));
        pipe_->InitBuffer(vbBuf_, KdaBwdRecomputeArch35::kBt * KdaBwdRecomputeArch35::kV * sizeof(QkType));
        pipe_->InitBuffer(betaBuf_, KdaBwdRecomputeArch35::kBt * sizeof(float));
        pipe_->InitBuffer(accBuf_, KdaBwdRecomputeArch35::kK * sizeof(float));
        pipe_->InitBuffer(dtBiasBuf_, KdaBwdRecomputeArch35::kK * sizeof(float));
        pipe_->InitBuffer(gkLastBuf_, KdaBwdRecomputeArch35::kK * sizeof(float));

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(chunkFreeFlag_);
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(chunkFreeFlag_);

        const uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
        const uint32_t coreNumAic = GetBlockNum();
        uint32_t vecTaskIdx = 0;

        for (uint32_t loopIdx = coreIdx; loopIdx < chunkNum_; loopIdx += coreNumAic) {
            const uint32_t slot = loopIdx & 1U;
            Catlass::Arch::CrossCoreWaitFlag(chunkFreeFlag_);

            uint32_t bos = 0;
            uint32_t eos = 0;
            KdaBwdRecomputeGetChunkOffset(
                cuSeqlens_, chunkIndices_, B_, Hv_, T_, chunkSize_, loopIdx, bos, eos);
            const uint32_t curChunkSize = eos - bos;
            const uint16_t validRows = static_cast<uint16_t>(curChunkSize);

            LocalTensor<float> gateFp32 = gateBuf_.Get<float>();
            LocalTensor<QkType> qLocal = qBuf_.Get<QkType>();
            LocalTensor<QkType> kLocal = kBuf_.Get<QkType>();
            LocalTensor<QkType> vLocal = vBuf_.Get<QkType>();
            LocalTensor<QkType> qgLocal = qgBuf_.Get<QkType>();
            LocalTensor<QkType> kbgLocal = kbgBuf_.Get<QkType>();
            LocalTensor<QkType> kgLocal = kgBuf_.Get<QkType>();
            LocalTensor<QkType> vbLocal = vbBuf_.Get<QkType>();
            LocalTensor<float> betaFp32 = betaBuf_.Get<float>();
            LocalTensor<float> accFp32 = accBuf_.Get<float>();
            LocalTensor<float> dtBias = dtBiasBuf_.Get<float>();
            LocalTensor<float> gkLast = gkLastBuf_.Get<float>();

            LocalTensor<QkType> kbgL1 = l1Buffer_[KdaBwdRecomputeArch35::KbgSlotOffset(slot)].template ReinterpretCast<QkType>();
            LocalTensor<QkType> vbL1 = l1Buffer_[KdaBwdRecomputeArch35::VbSlotOffset(slot)].template ReinterpretCast<QkType>();

            for (uint64_t h = 0; h < Hv_; ++h) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(chunkReadyFlag_);
                    continue;
                }

                const uint64_t hk = h / hvPerHk_;
                const uint64_t coreLoopsInB = (T_ + chunkSize_ - 1) / chunkSize_;
                const uint64_t bIdx = cuSeqlens_ ? 0 : (loopIdx / coreLoopsInB);
                const uint64_t bosK = cuSeqlens_ ? bos : (bos - bIdx * (Hv_ - Hk_) * T_);
                const uint64_t gBase = (h * T_ + bos) * KdaBwdRecomputeArch35::kK;
                const uint64_t qkBase = (hk * T_ + bosK) * KdaBwdRecomputeArch35::kK;
                const uint64_t outBase = (h * T_ + bos) * KdaBwdRecomputeArch35::kK;
                const uint64_t vBase = (h * T_ + bos) * KdaBwdRecomputeArch35::kV;
                const uint64_t betaBase = h * T_ + bos;

                if (hasDtBias_) {
                    DataCopy(dtBias, dtBiasTensor_[h * KdaBwdRecomputeArch35::kK], KdaBwdRecomputeArch35::kK);
                }
                Duplicate(accFp32, 0.0f, KdaBwdRecomputeArch35::kK);

                DataCopy(qLocal, qTensor_[qkBase], curChunkSize * KdaBwdRecomputeArch35::kK);
                DataCopy(kLocal, kTensor_[qkBase], curChunkSize * KdaBwdRecomputeArch35::kK);
                DataCopy(vLocal, vTensor_[vBase], curChunkSize * KdaBwdRecomputeArch35::kV);
                if constexpr (std::is_same<GateType, float>::value) {
                    DataCopy(gateFp32, gTensor_[gBase], curChunkSize * KdaBwdRecomputeArch35::kK);
                } else {
                    Cast(gateFp32, gTensor_[gBase], RoundMode::CAST_NONE, curChunkSize * KdaBwdRecomputeArch35::kK);
                }
                if constexpr (std::is_same<BetaType, float>::value) {
                    DataCopyPad(betaFp32, betaTensor_[betaBase],
                                {1, static_cast<uint32_t>(curChunkSize * sizeof(float)), 0, 0, 0},
                                {false, 0, 0, 0});
                } else {
                    LocalTensor<BetaType> betaRaw = betaRawBuf_.Get<BetaType>();
                    DataCopyPad(betaRaw, betaTensor_[betaBase],
                                {1, static_cast<uint32_t>(curChunkSize * sizeof(BetaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    Cast(betaFp32, betaRaw, RoundMode::CAST_NONE, curChunkSize);
                }
                PipeBarrier<PIPE_V>();

                if (useGate_) {
                    if (hasDtBias_) {
                        KdaBwdRecomputeArch35::AccumulateSafeGateChunk128Regbase<true>(
                            reinterpret_cast<__ubuf__ float *>(gateFp32.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ float *>(dtBias.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ float *>(accFp32.GetPhyAddr()),
                            validRows, expA_, lowerBound_);
                    } else {
                        KdaBwdRecomputeArch35::AccumulateSafeGateChunk128Regbase<false>(
                            reinterpret_cast<__ubuf__ float *>(gateFp32.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ float *>(dtBias.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ float *>(accFp32.GetPhyAddr()),
                            validRows, expA_, lowerBound_);
                    }
                } else {
                    for (uint32_t row = 0; row < curChunkSize; ++row) {
                        Add(accFp32, accFp32, gateFp32[row * KdaBwdRecomputeArch35::kK], KdaBwdRecomputeArch35::kK);
                        DataCopy(gateFp32[row * KdaBwdRecomputeArch35::kK], accFp32, KdaBwdRecomputeArch35::kK);
                    }
                    PipeBarrier<PIPE_V>();
                    Muls(gateFp32, gateFp32, KDA_BWD_RECOMPUTE_RCP_LN2, curChunkSize * KdaBwdRecomputeArch35::kK);
                }

                PipeBarrier<PIPE_V>();
                DataCopy(gkTensor_[outBase], gateFp32, curChunkSize * KdaBwdRecomputeArch35::kK);
                DataCopy(gkLast, gateFp32[(curChunkSize - 1) * KdaBwdRecomputeArch35::kK], KdaBwdRecomputeArch35::kK);

                KdaBwdRecomputeArch35::ComputeQgKbgKgRegbase<QkType, QkType>(
                    reinterpret_cast<__ubuf__ QkType *>(qLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ QkType *>(kLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ QkType *>(qgLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ QkType *>(kbgLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ QkType *>(kgLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(gateFp32.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(gkLast.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(betaFp32.GetPhyAddr()),
                    KdaBwdRecomputeArch35::kBt, KdaBwdRecomputeArch35::kK, validRows);

                KdaBwdRecomputeArch35::ComputeVbRegbase<QkType>(
                    reinterpret_cast<__ubuf__ QkType *>(vLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ QkType *>(vbLocal.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(betaFp32.GetPhyAddr()),
                    KdaBwdRecomputeArch35::kBt, KdaBwdRecomputeArch35::kV, validRows);

                PipeBarrier<PIPE_V>();
                DataCopy(qgTensor_[outBase], qgLocal, curChunkSize * KdaBwdRecomputeArch35::kK);
                DataCopy(kgTensor_[outBase], kgLocal, curChunkSize * KdaBwdRecomputeArch35::kK);

                DataCopy(kbgL1, kbgLocal, curChunkSize * KdaBwdRecomputeArch35::kK);
                DataCopy(vbL1, vbLocal, curChunkSize * KdaBwdRecomputeArch35::kV);
                PipeBarrier<PIPE_MTE3>();

                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(chunkReadyFlag_);
            }
        }
    }

private:
    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR v_;
    GM_ADDR g_;
    GM_ADDR beta_;
    GM_ADDR aLog_;
    GM_ADDR dtBias_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR gk_;
    GM_ADDR w_;
    GM_ADDR u_;
    GM_ADDR qg_;
    GM_ADDR kg_;
    GM_ADDR workspace_;
    const ChunkKdaBwdRecomputeTilingData *tiling_;
    TPipe *pipe_ = nullptr;

    GlobalTensor<QkType> qTensor_;
    GlobalTensor<QkType> kTensor_;
    GlobalTensor<QkType> vTensor_;
    GlobalTensor<GateType> gTensor_;
    GlobalTensor<BetaType> betaTensor_;
    GlobalTensor<float> aLogTensor_;
    GlobalTensor<float> dtBiasTensor_;
    GlobalTensor<float> gkTensor_;
    GlobalTensor<QkType> qgTensor_;
    GlobalTensor<QkType> kgTensor_;

    LocalTensor<uint8_t> l1Buffer_;
    Catlass::Arch::CrossCoreFlag chunkReadyFlag_;
    Catlass::Arch::CrossCoreFlag chunkFreeFlag_;

    TBuf<TPosition::VECCALC> betaRawBuf_;
    TBuf<TPosition::VECCALC> gateBuf_;
    TBuf<TPosition::VECCALC> qBuf_;
    TBuf<TPosition::VECCALC> kBuf_;
    TBuf<TPosition::VECCALC> vBuf_;
    TBuf<TPosition::VECCALC> qgBuf_;
    TBuf<TPosition::VECCALC> kbgBuf_;
    TBuf<TPosition::VECCALC> kgBuf_;
    TBuf<TPosition::VECCALC> vbBuf_;
    TBuf<TPosition::VECCALC> betaBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> dtBiasBuf_;
    TBuf<TPosition::VECCALC> gkLastBuf_;

    uint64_t B_ = 0;
    uint64_t Hk_ = 0;
    uint64_t Hv_ = 0;
    uint64_t hvPerHk_ = 1;
    uint64_t T_ = 0;
    uint64_t chunkNum_ = 0;
    uint64_t chunkSize_ = 64;
    bool useGate_ = true;
    bool hasDtBias_ = false;
    float expA_ = 1.0f;
    float lowerBound_ = -5.0f;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_RECOMPUTE_ARCH35_VECTOR_H
