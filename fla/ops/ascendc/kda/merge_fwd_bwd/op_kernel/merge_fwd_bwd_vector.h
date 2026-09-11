/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef MERGE_FWD_BWD_VECTOR_H
#define MERGE_FWD_BWD_VECTOR_H

#include "merge_fwd_bwd_common.h"

#include <type_traits>

using namespace AscendC;

namespace MergeFwBwd {

template <typename T>
class MergeFwdBwdVectorProcess {
public:
    __aicore__ inline MergeFwdBwdVectorProcess(
        GM_ADDR agHm, GM_ADDR h, GM_ADDR workspace, const MergeFwdBwdTilingData *tiling)
        : agHm_(agHm), h_(h), workspace_(workspace), tiling_(tiling)
    {
    }

    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe_ = pipe;
        hv_ = tiling_->Hv;
        s_ = tiling_->S;
        rank_ = tiling_->rank;
        n_ = tiling_->N;
        forward_ = tiling_->forward != 0;
        usedCoreNum_ = tiling_->usedAic > 0 ? static_cast<uint32_t>(tiling_->usedAic) : 1U;

        agHmTensor_.SetGlobalBuffer((__gm__ T *)agHm_);
        hTensor_.SetGlobalBuffer((__gm__ T *)h_);
        agHmTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        hTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        pipe_->InitBuffer(fp32Buf_, kKvElems * sizeof(float));
        pipe_->InitBuffer(hmmBuf_, kKvElems * sizeof(float));
        pipe_->InitBuffer(bf16Buf_, kKvElems * sizeof(bfloat16_t));
        mte2ToV_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte3_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte2ToMte3_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        mte3ToMte2_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
        const uint32_t subIdx = GetSubBlockIdx();
        uint64_t hBegin = 0;
        uint64_t hEnd = 0;
        CoreTaskRange(coreIdx, usedCoreNum_, static_cast<uint64_t>(hv_), hBegin, hEnd);
        for (uint64_t h = hBegin; h < hEnd; ++h) {
            if ((h & 1U) != subIdx) {
                continue;
            }
            ProcessHead(coreIdx, subIdx, static_cast<int64_t>(h));
        }
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToV_);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vToMte3_);
        pipe_->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2ToMte3_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

private:
    static constexpr uint32_t kTileElems = kTileM * kVDim;

    __aicore__ inline void CopyStridedToUb(LocalTensor<T> dst, int64_t srcRank, int64_t h, uint32_t col, uint32_t cols)
    {
        const uint64_t base = AgHmHeadOffset(srcRank, hv_, h) + col;
        DataCopyExtParams params;
        params.blockCount = static_cast<uint16_t>(kKDim);
        params.blockLen = static_cast<uint32_t>(cols * sizeof(T));
        params.srcStride = static_cast<uint32_t>((kRowStride - cols) * sizeof(T));
        params.dstStride = 0;
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, agHmTensor_[base], params, pad);
    }

    __aicore__ inline void LoadHeFp32(int64_t srcRank, int64_t h)
    {
        LocalTensor<float> dst = fp32Buf_.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            CopyStridedToUb(dst, srcRank, h, 0, kVDim);
            SetFlag<HardEvent::MTE2_V>(mte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
        } else {
            LocalTensor<T> src = bf16Buf_.Get<T>();
            CopyStridedToUb(src, srcRank, h, 0, kVDim);
            SetFlag<HardEvent::MTE2_V>(mte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
            Cast(dst, src, RoundMode::CAST_NONE, kKvElems);
        }
    }

    __aicore__ inline void StoreMBf16(int64_t srcRank, int64_t h, GlobalTensor<bfloat16_t> dstGm)
    {
        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<float> src = fp32Buf_.Get<float>();
            LocalTensor<bfloat16_t> dst = bf16Buf_.Get<bfloat16_t>();
            CopyStridedToUb(src, srcRank, h, kVDim, kKDim);
            SetFlag<HardEvent::MTE2_V>(mte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
            Cast(dst, src, RoundMode::CAST_RINT, kKkElems);
            SetFlag<HardEvent::V_MTE3>(vToMte3_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3_);
            DataCopy(dstGm, dst, kKkElems);
        } else {
            LocalTensor<bfloat16_t> dst = bf16Buf_.Get<bfloat16_t>();
            CopyStridedToUb(dst.template ReinterpretCast<T>(), srcRank, h, kVDim, kKDim);
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
            DataCopy(dstGm, dst, kKkElems);
        }
    }

    __aicore__ inline void StoreHBf16(GM_ADDR head)
    {
        LocalTensor<float> hFp32 = fp32Buf_.Get<float>();
        LocalTensor<bfloat16_t> dst = bf16Buf_.Get<bfloat16_t>();
        GlobalTensor<bfloat16_t> gmH;
        gmH.SetGlobalBuffer((__gm__ bfloat16_t *)head);
        gmH.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        Cast(dst, hFp32, RoundMode::CAST_RINT, kKvElems);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopy(gmH, dst, kKvElems);
    }

    __aicore__ inline void StoreUserH(int64_t h)
    {
        LocalTensor<float> hFp32 = fp32Buf_.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(vToMte3_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3_);
            DataCopy(hTensor_[static_cast<uint64_t>(h) * kKvElems], hFp32, kKvElems);
        } else {
            LocalTensor<T> dst = bf16Buf_.Get<T>();
            Cast(dst, hFp32, RoundMode::CAST_RINT, kKvElems);
            SetFlag<HardEvent::V_MTE3>(vToMte3_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3_);
            DataCopy(hTensor_[static_cast<uint64_t>(h) * kKvElems], dst, kKvElems);
        }
    }

    __aicore__ inline GM_ADDR HeadPtr(int64_t h) const
    {
        return h_ + static_cast<uint64_t>(h) * static_cast<uint64_t>(kKvElems) * sizeof(T);
    }

    __aicore__ inline void LoadCTileTo(LocalTensor<float> dst, int64_t h, uint32_t tile)
    {
        GlobalTensor<T> gmC;
        gmC.SetGlobalBuffer(
            (__gm__ T *)agHm_ + AgHmHeadOffset(rank_, hv_, h) +
            static_cast<uint64_t>(tile) * kTileElems);
        gmC.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if constexpr (std::is_same<T, float>::value) {
            DataCopy(dst, gmC, kTileElems);
            SetFlag<HardEvent::MTE2_V>(mte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
        } else {
            LocalTensor<T> src = bf16Buf_.Get<T>();
            DataCopy(src, gmC, kTileElems);
            SetFlag<HardEvent::MTE2_V>(mte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
            Cast(dst, src, RoundMode::CAST_NONE, kTileElems);
        }
    }

    // Vector Add/Adds ignore count when dst remaining size != count.
    // dst = tensor[8192] (remaining 8192) is the only Add dest that works.
    __aicore__ inline void AssembleTopFromHmmTail(int64_t h)
    {
        LocalTensor<float> hFp32 = fp32Buf_.Get<float>();
        LocalTensor<float> hmm = hmmBuf_.Get<float>();
        GlobalTensor<float> gm;
        gm.SetGlobalBuffer((__gm__ float *)HeadPtr(h));
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopy(gm, hmm[kTileElems], kTileElems);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        DataCopy(hFp32, gm, kTileElems);
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void StoreStageInputs(int64_t h, int64_t mSrc)
    {
        GM_ADDR head = HeadPtr(h);
        StoreHBf16(head);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        if constexpr (std::is_same<T, float>::value) {
            GlobalTensor<bfloat16_t> gmMBf16;
            gmMBf16.SetGlobalBuffer((__gm__ bfloat16_t *)(head + kHBf16Bytes));
            StoreMBf16(mSrc, h, gmMBf16);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        }
        (void)mSrc;
    }

    __aicore__ inline void ProcessHead(uint32_t coreIdx, uint32_t slot, int64_t h)
    {
        (void)coreIdx;
        LocalTensor<float> hFp32 = fp32Buf_.Get<float>();
        LocalTensor<float> hmm = hmmBuf_.Get<float>();

        const int64_t src0 = SrcRank(rank_, n_, 0, forward_, s_);
        if (n_ <= 1) {
            if constexpr (std::is_same<T, float>::value) {
                CopyStridedToUb(hFp32, src0, h, 0, kVDim);
                SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
                WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
                DataCopy(hTensor_[static_cast<uint64_t>(h) * kKvElems], hFp32, kKvElems);
            } else {
                LoadHeFp32(src0, h);
                StoreUserH(h);
            }
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            return;
        }

        LoadHeFp32(src0, h);
        StoreStageInputs(h, SrcRank(rank_, n_, 1, forward_, s_));
        AivSetChunkReady<PIPE_MTE3>();

        for (int64_t i = 1; i < n_; ++i) {
            AivWaitChunkFree<PIPE_MTE3>();
            AivSetChunkReady<PIPE_MTE3>();
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            LoadHeFp32(SrcRank(rank_, n_, i, forward_, s_), h);
            Adds(hmmBuf_.Get<float>()[kTileElems], fp32Buf_.Get<float>(), 0.0f, kTileElems);
            PipeBarrier<PIPE_V>();
            LoadCTileTo(fp32Buf_.Get<float>(), h, 0);
            Add(hmmBuf_.Get<float>()[kTileElems], fp32Buf_.Get<float>(),
                hmmBuf_.Get<float>()[kTileElems], kTileElems);
            PipeBarrier<PIPE_V>();
            AssembleTopFromHmmTail(h);
            AivWaitChunkFree<PIPE_MTE3>();
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            {
                LocalTensor<float> hmm2 = hmmBuf_.Get<float>();
                LocalTensor<float> h2 = fp32Buf_.Get<float>();
                LoadCTileTo(hmm2, h, 1);
                Add(h2[kTileElems], hmm2, h2[kTileElems], kTileElems);
            }
            if (i + 1 < n_) {
                StoreStageInputs(h, SrcRank(rank_, n_, i + 1, forward_, s_));
                AivSetChunkReady<PIPE_MTE3>();
            } else {
                StoreUserH(h);
                SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
                WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            }
        }
        (void)slot;
    }

    TPipe *pipe_ = nullptr;
    GM_ADDR agHm_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    const MergeFwdBwdTilingData *tiling_ = nullptr;
    GlobalTensor<T> agHmTensor_;
    GlobalTensor<T> hTensor_;
    TBuf<TPosition::VECCALC> fp32Buf_;
    TBuf<TPosition::VECCALC> hmmBuf_;
    TBuf<TPosition::VECCALC> bf16Buf_;
    int32_t mte2ToV_ = 0;
    int32_t vToMte3_ = 0;
    int32_t mte2ToMte3_ = 0;
    int32_t mte3ToMte2_ = 0;
    int64_t hv_ = 1;
    int64_t s_ = 1;
    int64_t rank_ = 0;
    int64_t n_ = 1;
    bool forward_ = true;
    uint32_t usedCoreNum_ = 1;
};

} // namespace MergeFwBwd

#endif
