/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef MERGE_FWD_BWD_STRUCT_H
#define MERGE_FWD_BWD_STRUCT_H

#include <cstdint>

namespace MergeFwBwd {

struct MergeFwdBwdTilingData {
    int64_t S;
    int64_t Hv;
    int64_t K;
    int64_t V;
    int64_t rank;
    int64_t N;
    int64_t forward;
    int64_t usedAic;
    int64_t sysWorkspaceSize;
};

static_assert(sizeof(MergeFwdBwdTilingData) == 72, "TilingData is 9 int64 fields");

constexpr uint32_t kKDim = 128;
constexpr uint32_t kVDim = 128;
constexpr uint32_t kTileM = 64;
constexpr uint32_t kRowStride = kVDim + kKDim; // 256
constexpr uint32_t kKvElems = kKDim * kVDim;
constexpr uint32_t kKkElems = kKDim * kKDim;

constexpr uint32_t kHBf16Bytes = kKvElems * 2;
constexpr uint32_t kMBf16Bytes = kKkElems * 2;
constexpr uint32_t kHmmBytes = kKvElems * 4;
constexpr uint32_t kSlotBytes = kHBf16Bytes + kMBf16Bytes + kHmmBytes; // 128 KiB
constexpr uint32_t kCoreStride = 2 * kSlotBytes;

constexpr uint8_t kCrossCoreModeIndep = 0x4;
constexpr uint8_t kChunkReadyFlag = 6;
constexpr uint8_t kChunkFreeFlag = 7;
constexpr uint8_t kSubBlockFlagOffset = 16;

} // namespace MergeFwBwd

#endif
