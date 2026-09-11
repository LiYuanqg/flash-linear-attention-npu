/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef MERGE_FWD_BWD_CUBE_H
#define MERGE_FWD_BWD_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include <type_traits>

#include "merge_fwd_bwd_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace tla;

namespace MergeFwBwd {

template <typename DstT>
class MergeFwdBwdCubeProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using CubeT = bfloat16_t;
    using AccT = float;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, CubeT, LayoutTagA, CubeT, LayoutTagB, AccT, LayoutTagC>;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, CubeT, LayoutTagL1A>;

    static constexpr int32_t kEventL1 = 0;
    static constexpr int32_t kEventL0A = 0;
    static constexpr int32_t kEventL0B = 1;
    static constexpr int32_t kEventL0C0 = 0;
    static constexpr int32_t kEventL0C1 = 1;
    static constexpr int32_t kEventMte1M = 0;
    static constexpr uint32_t kL0CTileBytes = 128 * 1024;
    // H 32KiB + two 64x128 A tiles with 32KiB padding each (NZ C0 stride may pad).
    static constexpr uint32_t kL1HOffset = 0;
    static constexpr uint32_t kL1A0Offset = kHBf16Bytes;
    static constexpr uint32_t kL1A1Offset = kHBf16Bytes + kHBf16Bytes;
    static constexpr uint32_t kL1SlotStride = kHBf16Bytes * 3;

    __aicore__ inline MergeFwdBwdCubeProcess(GM_ADDR agHm, GM_ADDR h) : agHm_(agHm), h_(h) {}

    __aicore__ inline void Init(const MergeFwdBwdTilingData *tiling)
    {
        hv_ = static_cast<uint64_t>(tiling->Hv);
        s_ = tiling->S;
        rank_ = tiling->rank;
        n_ = tiling->N;
        forward_ = tiling->forward != 0;
        usedCoreNum_ = tiling->usedAic > 0 ? static_cast<uint32_t>(tiling->usedAic) : 1U;
    }

    __aicore__ inline void Process()
    {
        if (n_ <= 1) {
            return;
        }
        Arch::Resource<ArchTag> resource;
        AscendC::LocalTensor<CubeT> l0A = resource.l0ABuf.template GetBufferByByte<CubeT>(0);
        AscendC::LocalTensor<CubeT> l0B = resource.l0BBuf.template GetBufferByByte<CubeT>(0);
        AscendC::LocalTensor<AccT> l0C0 = resource.l0CBuf.template GetBufferByByte<AccT>(0);
        AscendC::LocalTensor<AccT> l0C1 = resource.l0CBuf.template GetBufferByByte<AccT>(kL0CTileBytes);

        auto layoutL1A = tla::MakeLayout<CubeT, LayoutTagL1A>(kTileM, kKDim);
        auto layoutL1B = tla::MakeLayout<CubeT, LayoutTagL1B>(kKDim, kVDim);
        LayoutTagA tagMTile = LayoutTagA::template MakeLayout<CubeT>(kTileM, kKDim);
        LayoutTagB tagH = LayoutTagB::template MakeLayout<CubeT>(kKDim, kVDim);
        LayoutTagA tagMTileStrided(kTileM, kKDim, static_cast<typename LayoutTagA::LongIndex>(kRowStride));
        auto layoutMTile = MakeLayoutFromTag(tagMTile);
        auto layoutH = MakeLayoutFromTag(tagH);
        auto layoutMTileStrided = MakeLayoutFromTag(tagMTileStrided);
        auto layoutL0A = tla::MakeLayout<CubeT, LayoutTagL0A>(kTileM, kKDim);
        auto layoutL0B = tla::MakeLayout<CubeT, LayoutTagL0B>(kKDim, kVDim);
        auto layoutL0C = tla::MakeLayoutL0C(kTileM, kVDim);
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        uint64_t hBegin = 0;
        uint64_t hEnd = 0;
        CoreTaskRange(coreIdx, usedCoreNum_, hv_, hBegin, hEnd);

        for (uint64_t h = hBegin; h < hEnd; ++h) {
            const uint32_t slot = static_cast<uint32_t>(h & 1U);
            const uint32_t l1Base = slot * kL1SlotStride;
            AscendC::LocalTensor<CubeT> hL1 = resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1HOffset);
            AscendC::LocalTensor<CubeT> a0L1 = resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1A0Offset);
            AscendC::LocalTensor<CubeT> a1L1 = resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1A1Offset);
            auto tensorL1B = tla::MakeTensor(hL1, layoutL1B, Arch::PositionL1{});
            auto tensorL1A0 = tla::MakeTensor(a0L1, layoutL1A, Arch::PositionL1{});
            auto tensorL1A1 = tla::MakeTensor(a1L1, layoutL1A, Arch::PositionL1{});

            const int64_t gemmSteps = n_ - 1;
            for (int64_t step = 0; step < gemmSteps; ++step) {
                AicWaitChunkReady<PIPE_MTE2>(slot);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1);
                LoadHToL1(tensorL1B, layoutH, h);
                LoadATileToL1(tensorL1A0, layoutMTile, layoutMTileStrided, h, step + 1, 0);
                LoadATileToL1(tensorL1A1, layoutMTile, layoutMTileStrided, h, step + 1, kTileM);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEventL1);

                auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Arch::PositionL0A{});
                auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Arch::PositionL0B{});
                auto tensorL0C0 = tla::MakeTensor(l0C0, layoutL0C, Arch::PositionL0C{});
                auto tensorL0C1 = tla::MakeTensor(l0C1, layoutL0C, Arch::PositionL0C{});
                auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(kKDim, kVDim));
                auto tileL1A0 = GetTile(tensorL1A0, tla::MakeCoord(0, 0), tla::MakeShape(kTileM, kKDim));
                auto tileL1A1 = GetTile(tensorL1A1, tla::MakeCoord(0, 0), tla::MakeShape(kTileM, kKDim));

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
                copyL1ToL0A(tensorL0A, tileL1A0);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
                copyL1ToL0B(tensorL0B, tileL1B);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventMte1M);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventMte1M);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
                tileMmad(tensorL0C0, tensorL0A, tensorL0B, kTileM, kVDim, kKDim, true, 0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEventL0C0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEventL0C0);
                CopyCToAgHm(h, tensorL0C0, 0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);

                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
                AicSetChunkFree<PIPE_MTE1>(slot);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);

                AicWaitChunkReady<PIPE_MTE2>(slot);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
                copyL1ToL0A(tensorL0A, tileL1A1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventMte1M);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventMte1M);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
                tileMmad(tensorL0C1, tensorL0A, tensorL0B, kTileM, kVDim, kKDim, true, 0b11);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEventL0C1);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEventL0C1);
                CopyCToAgHm(h, tensorL0C1, 1);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
                AicSetChunkFree<PIPE_MTE1>(slot);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
    }

private:
    __aicore__ inline GM_ADDR HeadPtr(uint64_t h) const
    {
        return h_ + h * static_cast<uint64_t>(kKvElems) * sizeof(DstT);
    }

    template <typename TensorL1B, typename LayoutH>
    __aicore__ inline void LoadHToL1(TensorL1B const &tensorL1B, LayoutH const &layoutH, uint64_t h)
    {
        AscendC::GlobalTensor<CubeT> gmH;
        gmH.SetGlobalBuffer((__gm__ CubeT *)HeadPtr(h));
        gmH.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        auto tensorHGm = tla::MakeTensor(gmH, layoutH, Arch::PositionGM{});
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(tensorHGm)>;
        CopyGmToL1B copyGmToL1B;
        copyGmToL1B(tensorL1B, tensorHGm);
    }

    template <typename TensorL1A, typename LayoutM, typename LayoutMStrided>
    __aicore__ inline void LoadATileToL1(
        TensorL1A const &tensorL1A, LayoutM const &layoutM, LayoutMStrided const &layoutMStrided,
        uint64_t h, int64_t mIdx, uint32_t m0)
    {
        AscendC::GlobalTensor<CubeT> gmM;
        if constexpr (std::is_same<DstT, float>::value) {
            GM_ADDR head = HeadPtr(h);
            gmM.SetGlobalBuffer((__gm__ CubeT *)(head + kHBf16Bytes) + static_cast<uint64_t>(m0) * kKDim);
            auto tensorMGm = tla::MakeTensor(gmM, layoutM, Arch::PositionGM{});
            using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(tensorMGm)>;
            CopyGmToL1A copyGmToL1A;
            copyGmToL1A(tensorL1A, tensorMGm);
        } else {
            const int64_t srcM = SrcRank(rank_, n_, mIdx, forward_, s_);
            gmM.SetGlobalBuffer(
                (__gm__ CubeT *)agHm_ + AgHmHeadOffset(srcM, static_cast<int64_t>(hv_), static_cast<int64_t>(h)) +
                kVDim + static_cast<uint64_t>(m0) * kRowStride);
            auto tensorMGm = tla::MakeTensor(gmM, layoutMStrided, Arch::PositionGM{});
            using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(tensorMGm)>;
            CopyGmToL1A copyGmToL1A;
            copyGmToL1A(tensorL1A, tensorMGm);
        }
    }

    template <typename TensorL0C>
    __aicore__ inline void CopyCToAgHm(uint64_t h, TensorL0C const &tensorL0C, uint32_t tile)
    {
        const uint64_t elemOff = static_cast<uint64_t>(tile) * kTileM * kVDim;
        AscendC::GlobalTensor<DstT> gmC;
        gmC.SetGlobalBuffer(
            (__gm__ DstT *)agHm_ +
            AgHmHeadOffset(rank_, static_cast<int64_t>(hv_), static_cast<int64_t>(h)) +
            elemOff);
        gmC.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        CopyL0CTile(gmC, tensorL0C, kVDim);
    }

    template <typename TensorL0C>
    __aicore__ inline void CopyL0CTile(
        AscendC::GlobalTensor<DstT> const &gmC, TensorL0C const &tensorL0C, uint32_t dstStride)
    {
        AscendC::DataCopyCO12DstParams params;
        params.nSize = kVDim;
        params.mSize = kTileM;
        params.dstStride = dstStride;
        params.srcStride = tla::get<1, 1>(tensorL0C.stride()) / tla::get<0, 0>(tensorL0C.stride());
        params.quantPre = Gemm::Tile::CopyL0CToDstQuantMode<
            ArchTag, AccT, DstT, Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;
        params.nz2ndEn = true;
        params.reluPre = false;
        params.unitFlag = 0b11;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        auto srcOffset = tensorL0C.layout()(tensorL0C.coord());
        AscendC::DataCopy(gmC, tensorL0C.data()[srcOffset], params);
    }

    GM_ADDR agHm_ = nullptr;
    GM_ADDR h_ = nullptr;
    uint64_t hv_ = 1;
    int64_t s_ = 1;
    int64_t rank_ = 0;
    int64_t n_ = 1;
    bool forward_ = true;
    uint32_t usedCoreNum_ = 1;
};

} // namespace MergeFwBwd

#endif
