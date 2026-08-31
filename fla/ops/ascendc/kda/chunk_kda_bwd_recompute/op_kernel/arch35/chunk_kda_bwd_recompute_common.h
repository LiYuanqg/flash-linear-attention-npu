/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH35_COMMON_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH35_COMMON_H

#include "../chunk_kda_bwd_recompute_common.h"

namespace KdaBwdRecomputeArch35 {

constexpr uint32_t kBt = 64;
constexpr uint32_t kK = 128;
constexpr uint32_t kV = 128;
constexpr uint32_t kBk = 64;
constexpr uint32_t kBv = 64;

constexpr uint32_t kL1ABytes = 8 * 1024;
constexpr uint32_t kL1SlotRegionBytes = 32 * 1024;
constexpr uint32_t kL1VectorBytes = kBt * kK * sizeof(uint16_t);

constexpr uint32_t kL1AOffset = 0;
constexpr uint32_t kL1KbgSlot0Offset = 8 * 1024;
constexpr uint32_t kL1KbgSlot1Offset = 40 * 1024;
constexpr uint32_t kL1VbSlot0Offset = 72 * 1024;
constexpr uint32_t kL1VbSlot1Offset = 104 * 1024;

constexpr uint8_t kChunkReadyFlag = 6;
constexpr uint8_t kChunkFreeFlag = 7;

__aicore__ inline uint32_t KbgSlotOffset(uint32_t slot)
{
    return slot == 0 ? kL1KbgSlot0Offset : kL1KbgSlot1Offset;
}

__aicore__ inline uint32_t VbSlotOffset(uint32_t slot)
{
    return slot == 0 ? kL1VbSlot0Offset : kL1VbSlot1Offset;
}

} // namespace KdaBwdRecomputeArch35

#endif // CHUNK_KDA_BWD_RECOMPUTE_ARCH35_COMMON_H
