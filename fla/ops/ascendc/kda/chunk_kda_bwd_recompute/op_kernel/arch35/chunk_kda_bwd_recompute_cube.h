/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH35_CUBE_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH35_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "../chunk_kda_bwd_recompute_struct.h"
#include "../chunk_kda_bwd_recompute_common.h"
#include "chunk_kda_bwd_recompute_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace tla;

namespace KDA {

template <typename QkType>
class ChunkKdaBwdRecomputeCubeProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;

    __aicore__ inline ChunkKdaBwdRecomputeCubeProcess(
        GM_ADDR a, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace)
        : a_(a), cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices), w_(w), u_(u), workspace_(workspace)
    {
        (void)workspace_;
    }

    __aicore__ inline void Init(const ChunkKdaBwdRecomputeTilingData &tiling)
    {
        B_ = static_cast<uint64_t>(tiling.B);
        Hv_ = static_cast<uint64_t>(tiling.Hv);
        T_ = static_cast<uint64_t>(tiling.T);
        K_ = static_cast<uint64_t>(tiling.K);
        V_ = static_cast<uint64_t>(tiling.V);
        chunkNum_ = static_cast<uint64_t>(tiling.chunkNum);
        chunkSize_ = static_cast<uint64_t>(tiling.chunkSize);
        chunkReadyFlag_ = Catlass::Arch::CrossCoreFlag(KdaBwdRecomputeArch35::kChunkReadyFlag);
        chunkFreeFlag_ = Catlass::Arch::CrossCoreFlag(KdaBwdRecomputeArch35::kChunkFreeFlag);
    }

    __aicore__ inline void Process()
    {
        using LayoutTagA = layout::RowMajor;
        using LayoutTagB = layout::RowMajor;
        using LayoutTagC = layout::RowMajor;
        using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
        using L1TileShape = Shape<_128, _128, _256>;
        using L0TileShape = Shape<_128, _128, _128>;
        using TileCopy =
            Gemm::Tile::PackedTileCopyTla<ArchTag, QkType, LayoutTagA, QkType, LayoutTagB, QkType, LayoutTagC>;
        using BlockMmad =
            Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, QkType, QkType, QkType, void, TileCopy>;

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        AscendC::LocalTensor<QkType> aL1 =
            resource.l1Buf.template GetBufferByByte<QkType>(KdaBwdRecomputeArch35::kL1AOffset);

        for (uint32_t loopIdx = coreIdx; loopIdx < chunkNum_; loopIdx += AscendC::GetBlockNum()) {
            const uint32_t slot = loopIdx & 1U;
            uint32_t bos = 0;
            uint32_t eos = 0;
            KdaBwdRecomputeGetChunkOffset(
                cuSeqlens_, chunkIndices_, B_, Hv_, T_, chunkSize_, loopIdx, bos, eos);
            const uint32_t curChunkSize = eos - bos;

            AscendC::LocalTensor<QkType> kbgL1 = resource.l1Buf.template GetBufferByByte<QkType>(
                KdaBwdRecomputeArch35::KbgSlotOffset(slot));
            AscendC::LocalTensor<QkType> vbL1 = resource.l1Buf.template GetBufferByByte<QkType>(
                KdaBwdRecomputeArch35::VbSlotOffset(slot));

            for (uint64_t h = 0; h < Hv_; ++h) {
                Catlass::Arch::CrossCoreWaitFlag(chunkReadyFlag_);

                GlobalTensor<QkType> gmA;
                gmA.SetGlobalBuffer((__gm__ QkType *)a_ + (h * T_ + bos) * chunkSize_);
                DataCopy(aL1, gmA, curChunkSize * curChunkSize);
                PipeBarrier<PIPE_MTE2>();

                GlobalTensor<QkType> gmU;
                GlobalTensor<QkType> gmW;
                gmU.SetGlobalBuffer((__gm__ QkType *)u_ + (h * T_ + bos) * V_);
                gmW.SetGlobalBuffer((__gm__ QkType *)w_ + (h * T_ + bos) * K_);

                auto layoutA = LayoutTagA::template MakeLayout<QkType>(chunkSize_, chunkSize_);
                auto layoutVb = LayoutTagB::template MakeLayout<QkType>(chunkSize_, V_);
                auto layoutKbg = LayoutTagB::template MakeLayout<QkType>(chunkSize_, K_);
                auto layoutU = LayoutTagC::template MakeLayout<QkType>(chunkSize_, V_);
                auto layoutW = LayoutTagC::template MakeLayout<QkType>(chunkSize_, K_);

                auto tensorA = tla::MakeTensor(aL1, layoutA, Arch::PositionL1{});
                auto tensorVb = tla::MakeTensor(vbL1, layoutVb, Arch::PositionL1{});
                auto tensorKbg = tla::MakeTensor(kbgL1, layoutKbg, Arch::PositionL1{});
                auto tensorU = tla::MakeTensor(gmU, layoutU, Arch::PositionGM{});
                auto tensorW = tla::MakeTensor(gmW, layoutW, Arch::PositionGM{});

                GemmCoord shapeU{curChunkSize, static_cast<uint32_t>(V_), curChunkSize};
                auto blockA = GetTile(tensorA, MakeCoord(0, 0), MakeShape(shapeU.m(), shapeU.k()));
                auto blockVb = GetTile(tensorVb, MakeCoord(0, 0), MakeShape(shapeU.k(), shapeU.n()));
                auto blockU = GetTile(tensorU, MakeCoord(0, 0), MakeShape(shapeU.m(), shapeU.n()));
                blockMmad(blockA, blockVb, blockU, shapeU);

                GemmCoord shapeW{curChunkSize, static_cast<uint32_t>(K_), curChunkSize};
                auto blockKbg = GetTile(tensorKbg, MakeCoord(0, 0), MakeShape(shapeW.k(), shapeW.n()));
                auto blockW = GetTile(tensorW, MakeCoord(0, 0), MakeShape(shapeW.m(), shapeW.n()));
                blockMmad(blockA, blockKbg, blockW, shapeW);

                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(chunkFreeFlag_);
            }
        }
    }

private:
    GM_ADDR a_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR w_;
    GM_ADDR u_;
    GM_ADDR workspace_;
    uint64_t B_ = 0;
    uint64_t Hv_ = 0;
    uint64_t T_ = 0;
    uint64_t K_ = 128;
    uint64_t V_ = 128;
    uint64_t chunkNum_ = 0;
    uint64_t chunkSize_ = 64;
    Catlass::Arch::CrossCoreFlag chunkReadyFlag_;
    Catlass::Arch::CrossCoreFlag chunkFreeFlag_;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_RECOMPUTE_ARCH35_CUBE_H
