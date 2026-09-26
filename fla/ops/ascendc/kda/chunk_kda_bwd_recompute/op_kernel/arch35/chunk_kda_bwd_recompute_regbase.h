/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH35_REGBASE_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH35_REGBASE_H

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310

#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "kernel_utils/vector/regbase.hpp"
#endif

#include "chunk_kda_bwd_recompute_common.h"

namespace KdaBwdRecomputeArch35 {

constexpr float kLn2 = 0.69314718055994530942f;
constexpr float kExpInputMax = 80.0f * kLn2;
constexpr float kExpInputMin = -80.0f * kLn2;
// FwdPrepare V6 stores gk in log2 and clamps ±80 before *ln2 + fp32 Exp.
constexpr float kStoredExpMax = 80.0f;
constexpr float kStoredExpMin = -80.0f;

template <bool HAS_BIAS, bool HAS_ALOG>
static __simd_vf__ inline void AccumulateSafeGateChunk128Regbase(
    __ubuf__ float *input, __ubuf__ float *bias, __ubuf__ float *acc,
    uint16_t rows, __ubuf__ float *aLog, float lowerBound)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t ROW_ELEMENTS = 2 * FLOAT_ELEMENTS_PER_REG;

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> accZeroReg;
    RegTensor<float> accOneReg;
    RegTensor<float> oneZeroReg;
    RegTensor<float> oneOneReg;
    RegTensor<float> biasZeroReg;
    RegTensor<float> biasOneReg;
    RegTensor<float> expAReg;
    LoadAlign<float, LoadDist::DIST_NORM>(accZeroReg, acc);
    LoadAlign<float, LoadDist::DIST_NORM>(accOneReg, acc + FLOAT_ELEMENTS_PER_REG);
    Duplicate(oneZeroReg, 1.0f, floatMask);
    Duplicate(oneOneReg, 1.0f, floatMask);
    if constexpr (HAS_BIAS) {
        LoadAlign<float, LoadDist::DIST_NORM>(biasZeroReg, bias);
        LoadAlign<float, LoadDist::DIST_NORM>(biasOneReg, bias + FLOAT_ELEMENTS_PER_REG);
    }
    if constexpr (HAS_ALOG) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(expAReg, aLog);
        Exp(expAReg, expAReg, floatMask);
        Muls(expAReg, expAReg, -1.0f, floatMask);
    } else {
        Duplicate(expAReg, -1.0f, floatMask);
    }

    RegTensor<float> gateZeroReg;
    RegTensor<float> gateOneReg;
    RegTensor<float> sigmoidZeroReg;
    RegTensor<float> sigmoidOneReg;
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * ROW_ELEMENTS;
        LoadAlign<float, LoadDist::DIST_NORM>(gateZeroReg, input + rowOffset);
        LoadAlign<float, LoadDist::DIST_NORM>(gateOneReg, input + rowOffset + FLOAT_ELEMENTS_PER_REG);
        if constexpr (HAS_BIAS) {
            Add(gateZeroReg, gateZeroReg, biasZeroReg, floatMask);
            Add(gateOneReg, gateOneReg, biasOneReg, floatMask);
        }
        Mul(gateZeroReg, gateZeroReg, expAReg, floatMask);
        Mul(gateOneReg, gateOneReg, expAReg, floatMask);
        Exp(gateZeroReg, gateZeroReg, floatMask);
        Exp(gateOneReg, gateOneReg, floatMask);
        Adds(gateZeroReg, gateZeroReg, 1.0f, floatMask);
        Adds(gateOneReg, gateOneReg, 1.0f, floatMask);
        Div(sigmoidZeroReg, oneZeroReg, gateZeroReg, floatMask);
        Div(sigmoidOneReg, oneOneReg, gateOneReg, floatMask);
        Muls(sigmoidZeroReg, sigmoidZeroReg, lowerBound, floatMask);
        Muls(sigmoidOneReg, sigmoidOneReg, lowerBound, floatMask);
        Muls(sigmoidZeroReg, sigmoidZeroReg, KDA_BWD_RECOMPUTE_RCP_LN2, floatMask);
        Muls(sigmoidOneReg, sigmoidOneReg, KDA_BWD_RECOMPUTE_RCP_LN2, floatMask);
        Add(accZeroReg, accZeroReg, sigmoidZeroReg, floatMask);
        Add(accOneReg, accOneReg, sigmoidOneReg, floatMask);
        StoreAlign(input + rowOffset, accZeroReg, floatMask);
        StoreAlign(input + rowOffset + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
    }
    StoreAlign(acc, accZeroReg, floatMask);
    StoreAlign(acc + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
}

// FwdPrepare V0/V6 pairing: even/odd 64-lane regs, not K-contiguous [0:63]|[64:127].
// Load: float DIST_DINTLV_B32; bf16 LoadIn + CastHalf2Float ZERO/ONE.
// Store gk: DIST_INTLV_B32 so GM stays K-contiguous. Store bf16: CastFloat2Half packed.
template <typename InputT>
__simd_callee__ inline void LoadV0Pair(
    AscendC::MicroAPI::RegTensor<float> &lowReg,
    AscendC::MicroAPI::RegTensor<float> &highReg,
    __ubuf__ InputT *src,
    AscendC::MicroAPI::RegTensor<InputT> &packedReg)
{
    using namespace AscendC::MicroAPI;
    if constexpr (std::is_same<InputT, float>::value) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(lowReg, highReg, src);
        (void)packedReg;
    } else {
        MaskReg packedMask = CreateMask<InputT, MaskPattern::ALL>();
        LoadIn<InputT, false>(packedReg, src);
        CastHalf2Float<InputT>(lowReg, highReg, packedReg, packedMask);
    }
}

__simd_callee__ inline void StoreV0Gk(
    __ubuf__ float *dst,
    AscendC::MicroAPI::RegTensor<float> &lowReg,
    AscendC::MicroAPI::RegTensor<float> &highReg,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, lowReg, highReg, floatMask);
}

template <typename OutputT>
__simd_callee__ inline void StoreV6Packed(
    __ubuf__ OutputT *dst,
    AscendC::MicroAPI::RegTensor<float> &lowReg,
    AscendC::MicroAPI::RegTensor<float> &highReg,
    AscendC::MicroAPI::MaskReg &floatMask,
    AscendC::MicroAPI::RegTensor<OutputT> &packedReg)
{
    using namespace AscendC::MicroAPI;
    if constexpr (std::is_same<OutputT, float>::value) {
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, lowReg, highReg, floatMask);
        (void)packedReg;
    } else {
        MaskReg packedMask = CreateMask<OutputT, MaskPattern::ALL>();
        CastFloat2Half(packedReg, lowReg, highReg, floatMask);
        StoreAlign(dst, packedReg, packedMask);
    }
}

// Fallback K-contiguous pair for non-fused paths: regs hold [0:63] and [64:127].
template <typename InputT>
__simd_callee__ inline void LoadGateRegbasePair(
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    __ubuf__ InputT *src,
    AscendC::MicroAPI::MaskReg &inputMask,
    AscendC::MicroAPI::RegTensor<InputT> &inputReg)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    if constexpr (std::is_same<InputT, float>()) {
        LoadAlign<float, LoadDist::DIST_NORM>(zeroReg, src);
        LoadAlign<float, LoadDist::DIST_NORM>(oneReg, src + FLOAT_ELEMENTS_PER_REG);
        (void)inputReg;
        (void)inputMask;
        (void)floatMask;
    } else {
        LoadAlign<InputT, LoadDist::DIST_UNPACK_B16>(inputReg, src);
        Cast<float, InputT, ctHalf2Fp32Zero>(zeroReg, inputReg, floatMask);
        LoadAlign<InputT, LoadDist::DIST_UNPACK_B16>(inputReg, src + FLOAT_ELEMENTS_PER_REG);
        Cast<float, InputT, ctHalf2Fp32Zero>(oneReg, inputReg, floatMask);
        (void)inputMask;
    }
}

template <typename OutputT>
__simd_callee__ inline void StoreGateRegbasePair(
    __ubuf__ OutputT *dst,
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &inputMask,
    AscendC::MicroAPI::MaskReg &floatMask,
    AscendC::MicroAPI::RegTensor<OutputT> &outputReg)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    if constexpr (std::is_same<OutputT, float>()) {
        StoreAlign(dst, zeroReg, floatMask);
        StoreAlign(dst + FLOAT_ELEMENTS_PER_REG, oneReg, floatMask);
        (void)inputMask;
        (void)outputReg;
    } else {
        Cast<OutputT, float, ctFp322HalfZero>(outputReg, zeroReg, floatMask);
        StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(dst, outputReg, floatMask);
        Cast<OutputT, float, ctFp322HalfZero>(outputReg, oneReg, floatMask);
        StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(dst + FLOAT_ELEMENTS_PER_REG, outputReg, floatMask);
        (void)inputMask;
    }
}

// FwdPrepare V6 / leftover Repair: clamp stored log2 ±80, *ln2, fp32 Exp.
// Half Exp saturates at ~±11 (natural), which clips kg when |Glast-g| > 16.
__simd_callee__ inline void ExpPairStoredLog2(
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    Maxs(zeroReg, zeroReg, kStoredExpMin, floatMask);
    Maxs(oneReg, oneReg, kStoredExpMin, floatMask);
    Mins(zeroReg, zeroReg, kStoredExpMax, floatMask);
    Mins(oneReg, oneReg, kStoredExpMax, floatMask);
    Muls(zeroReg, zeroReg, kLn2, floatMask);
    Muls(oneReg, oneReg, kLn2, floatMask);
    Exp(zeroReg, zeroReg, floatMask);
    Exp(oneReg, oneReg, floatMask);
}

// Full-chunk VF: pass-1 matches FwdPrepare V0 even/odd DINTLV + two-step
// log2 scale + INTLV store; pass-2 matches V6 DINTLV load, log2 clamp ±80,
// *ln2, fp32 Exp, CastFloat2Half store. leftover must not enter this VF.
template <typename InputT, typename OutputT, typename GateT, typename BetaT, bool HAS_BIAS, bool HAS_ALOG,
          bool kFixed64 = false>
static __simd_vf__ inline void FusedRecomputeChunk128Regbase(
    __ubuf__ float *gk, __ubuf__ GateT *gIn, __ubuf__ float *bias, __ubuf__ float *aLog, __ubuf__ BetaT *betaRow,
    __ubuf__ InputT *q, __ubuf__ InputT *k, __ubuf__ InputT *v,
    __ubuf__ OutputT *qg, __ubuf__ OutputT *kbg, __ubuf__ OutputT *kg, __ubuf__ OutputT *vb,
    uint16_t rows, float lowerBound)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ROW_ELEMENTS = kK;
    const uint16_t nRows = kFixed64 ? static_cast<uint16_t>(64) : rows;

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg betaMask = CreateMask<BetaT, MaskPattern::ALL>();
    RegTensor<float> lastLowReg;
    RegTensor<float> lastHighReg;
    RegTensor<float> gateLowReg;
    RegTensor<float> gateHighReg;
    {
        RegTensor<float> accLowReg;
        RegTensor<float> accHighReg;
        RegTensor<float> oneLowReg;
        RegTensor<float> oneHighReg;
        RegTensor<float> biasLowReg;
        RegTensor<float> biasHighReg;
        RegTensor<float> expAReg;
        RegTensor<float> sigmoidLowReg;
        RegTensor<float> sigmoidHighReg;
        RegTensor<GateT> gatePackedReg;
        Duplicate(accLowReg, 0.0f, floatMask);
        Duplicate(accHighReg, 0.0f, floatMask);
        Duplicate(oneLowReg, 1.0f, floatMask);
        Duplicate(oneHighReg, 1.0f, floatMask);
        if constexpr (HAS_BIAS) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(biasLowReg, biasHighReg, bias);
        }
        if constexpr (HAS_ALOG) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(expAReg, aLog);
            Exp(expAReg, expAReg, floatMask);
        } else {
            Duplicate(expAReg, 1.0f, floatMask);
        }

        for (uint16_t row = 0; row < nRows; ++row) {
            const uint32_t rowOffset = static_cast<uint32_t>(row) * ROW_ELEMENTS;
            if constexpr (std::is_same<GateT, float>::value) {
                LoadV0Pair<float>(gateLowReg, gateHighReg, gk + rowOffset, gatePackedReg);
            } else {
                LoadV0Pair<GateT>(gateLowReg, gateHighReg, gIn + rowOffset, gatePackedReg);
            }
            if constexpr (HAS_BIAS) {
                Add(gateLowReg, gateLowReg, biasLowReg, floatMask);
                Add(gateHighReg, gateHighReg, biasHighReg, floatMask);
            }
            Mul(gateLowReg, gateLowReg, expAReg, floatMask);
            Mul(gateHighReg, gateHighReg, expAReg, floatMask);
            Muls(gateLowReg, gateLowReg, -1.0f, floatMask);
            Muls(gateHighReg, gateHighReg, -1.0f, floatMask);
            Exp(gateLowReg, gateLowReg, floatMask);
            Exp(gateHighReg, gateHighReg, floatMask);
            Adds(gateLowReg, gateLowReg, 1.0f, floatMask);
            Adds(gateHighReg, gateHighReg, 1.0f, floatMask);
            Div(sigmoidLowReg, oneLowReg, gateLowReg, floatMask);
            Div(sigmoidHighReg, oneHighReg, gateHighReg, floatMask);
            Muls(sigmoidLowReg, sigmoidLowReg, lowerBound, floatMask);
            Muls(sigmoidHighReg, sigmoidHighReg, lowerBound, floatMask);
            Muls(sigmoidLowReg, sigmoidLowReg, KDA_BWD_RECOMPUTE_RCP_LN2, floatMask);
            Muls(sigmoidHighReg, sigmoidHighReg, KDA_BWD_RECOMPUTE_RCP_LN2, floatMask);
            Add(accLowReg, accLowReg, sigmoidLowReg, floatMask);
            Add(accHighReg, accHighReg, sigmoidHighReg, floatMask);
            StoreV0Gk(gk + rowOffset, accLowReg, accHighReg, floatMask);
        }
        // last stays even/odd in registers, matching V0 gLast DIST_NORM of
        // DINTLV lanes. Do not reload last from the INTLV gk row.
        Adds(lastLowReg, accLowReg, 0.0f, floatMask);
        Adds(lastHighReg, accHighReg, 0.0f, floatMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    RegTensor<float> betaReg;
    RegTensor<BetaT> betaRawReg;
    RegTensor<float> expLowReg;
    RegTensor<float> expHighReg;
    RegTensor<float> qLowReg;
    RegTensor<float> qHighReg;
    RegTensor<float> kLowReg;
    RegTensor<float> kHighReg;
    RegTensor<float> outLowReg;
    RegTensor<float> outHighReg;
    RegTensor<float> deltaLowReg;
    RegTensor<float> deltaHighReg;
    RegTensor<float> vLowReg;
    RegTensor<float> vHighReg;
    RegTensor<InputT> inputReg;
    RegTensor<OutputT> outputReg;
    RegTensor<float> gkPackedReg;
    for (uint16_t row = 0; row < nRows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * static_cast<uint32_t>(kK);
        if constexpr (std::is_same<BetaT, float>::value) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, betaRow + row);
        } else {
            LoadIn<BetaT, true>(betaRawReg, betaRow + row);
            HalfOrFloat2Float(betaReg, betaRawReg, betaMask, floatMask);
        }

        LoadV0Pair<float>(gateLowReg, gateHighReg, gk + rowOffset, gkPackedReg);
        LoadV0Pair<InputT>(qLowReg, qHighReg, q + rowOffset, inputReg);
        LoadV0Pair<InputT>(kLowReg, kHighReg, k + rowOffset, inputReg);
        LoadV0Pair<InputT>(vLowReg, vHighReg, v + rowOffset, inputReg);
        Adds(expLowReg, gateLowReg, 0.0f, floatMask);
        Adds(expHighReg, gateHighReg, 0.0f, floatMask);
        ExpPairStoredLog2(expLowReg, expHighReg, floatMask);

        Mul(outLowReg, qLowReg, expLowReg, floatMask);
        Mul(outHighReg, qHighReg, expHighReg, floatMask);
        StoreV6Packed<OutputT>(qg + rowOffset, outLowReg, outHighReg, floatMask, outputReg);

        Mul(outLowReg, kLowReg, expLowReg, floatMask);
        Mul(outHighReg, kHighReg, expHighReg, floatMask);
        Mul(outLowReg, outLowReg, betaReg, floatMask);
        Mul(outHighReg, outHighReg, betaReg, floatMask);
        StoreV6Packed<OutputT>(kbg + rowOffset, outLowReg, outHighReg, floatMask, outputReg);

        Sub(deltaLowReg, lastLowReg, gateLowReg, floatMask);
        Sub(deltaHighReg, lastHighReg, gateHighReg, floatMask);
        ExpPairStoredLog2(deltaLowReg, deltaHighReg, floatMask);
        Mul(outLowReg, kLowReg, deltaLowReg, floatMask);
        Mul(outHighReg, kHighReg, deltaHighReg, floatMask);
        StoreV6Packed<OutputT>(kg + rowOffset, outLowReg, outHighReg, floatMask, outputReg);

        Mul(vLowReg, vLowReg, betaReg, floatMask);
        Mul(vHighReg, vHighReg, betaReg, floatMask);
        StoreV6Packed<OutputT>(vb + rowOffset, vLowReg, vHighReg, floatMask, outputReg);
    }
}

template <typename InputT, typename OutputT>
static __simd_vf__ inline void ComputeQgKbgKgRegbase(
    __ubuf__ InputT *q, __ubuf__ InputT *k, __ubuf__ OutputT *qg, __ubuf__ OutputT *kbg,
    __ubuf__ OutputT *kg, __ubuf__ float *gk, __ubuf__ float *gkLast, __ubuf__ float *betaRow,
    uint16_t rows, uint16_t cols, uint16_t validRows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(InputT);

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> lastZeroReg;
    RegTensor<float> lastOneReg;
    RegTensor<float> betaReg;
    RegTensor<float> gateZeroReg;
    RegTensor<float> gateOneReg;
    RegTensor<float> expZeroReg;
    RegTensor<float> expOneReg;
    RegTensor<float> qZeroReg;
    RegTensor<float> qOneReg;
    RegTensor<float> kZeroReg;
    RegTensor<float> kOneReg;
    RegTensor<float> outZeroReg;
    RegTensor<float> outOneReg;
    RegTensor<float> deltaZeroReg;
    RegTensor<float> deltaOneReg;
    RegTensor<float> floatScratchReg;
    RegTensor<InputT> inputReg;
    RegTensor<OutputT> outputReg;
    LoadAlign<float, LoadDist::DIST_NORM>(lastZeroReg, gkLast);
    LoadAlign<float, LoadDist::DIST_NORM>(
        lastOneReg, gkLast + (AscendC::VECTOR_REG_WIDTH / sizeof(float)));
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, betaRow + row);
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<InputT>(activeCount);
            const uint32_t offset = rowOffset + col;

            LoadGateRegbasePair<float>(gateZeroReg, gateOneReg, gk + offset, inputMask, floatScratchReg);
            Muls(expZeroReg, gateZeroReg, kLn2, floatMask);
            Muls(expOneReg, gateOneReg, kLn2, floatMask);
            Mins(expZeroReg, expZeroReg, kExpInputMax, floatMask);
            Mins(expOneReg, expOneReg, kExpInputMax, floatMask);
            Maxs(expZeroReg, expZeroReg, kExpInputMin, floatMask);
            Maxs(expOneReg, expOneReg, kExpInputMin, floatMask);
            Exp(expZeroReg, expZeroReg, floatMask);
            Exp(expOneReg, expOneReg, floatMask);

            LoadGateRegbasePair<InputT>(qZeroReg, qOneReg, q + offset, inputMask, inputReg);
            Mul(outZeroReg, qZeroReg, expZeroReg, floatMask);
            Mul(outOneReg, qOneReg, expOneReg, floatMask);
            StoreGateRegbasePair<OutputT>(qg + offset, outZeroReg, outOneReg, inputMask, floatMask, outputReg);

            LoadGateRegbasePair<InputT>(kZeroReg, kOneReg, k + offset, inputMask, inputReg);
            Mul(outZeroReg, kZeroReg, expZeroReg, floatMask);
            Mul(outOneReg, kOneReg, expOneReg, floatMask);
            Mul(outZeroReg, outZeroReg, betaReg, floatMask);
            Mul(outOneReg, outOneReg, betaReg, floatMask);
            StoreGateRegbasePair<OutputT>(kbg + offset, outZeroReg, outOneReg, inputMask, floatMask, outputReg);

            Sub(deltaZeroReg, lastZeroReg, gateZeroReg, floatMask);
            Sub(deltaOneReg, lastOneReg, gateOneReg, floatMask);
            Muls(deltaZeroReg, deltaZeroReg, kLn2, floatMask);
            Muls(deltaOneReg, deltaOneReg, kLn2, floatMask);
            Exp(deltaZeroReg, deltaZeroReg, floatMask);
            Exp(deltaOneReg, deltaOneReg, floatMask);
            Mul(outZeroReg, kZeroReg, deltaZeroReg, floatMask);
            Mul(outOneReg, kOneReg, deltaOneReg, floatMask);
            if (row >= validRows) {
                Duplicate(outZeroReg, 0.0f, floatMask);
                Duplicate(outOneReg, 0.0f, floatMask);
            }
            StoreGateRegbasePair<OutputT>(kg + offset, outZeroReg, outOneReg, inputMask, floatMask, outputReg);
        }
    }
}

template <typename VType>
static __simd_vf__ inline void ComputeVbRegbase(
    __ubuf__ VType *vIn, __ubuf__ VType *vbOut, __ubuf__ float *betaRow,
    uint16_t rows, uint16_t cols, uint16_t validRows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(VType);
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> betaReg;
    RegTensor<float> vZeroReg;
    RegTensor<float> vOneReg;
    RegTensor<VType> outReg;
    RegTensor<VType> inputReg;

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, betaRow + row);
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<VType>(activeCount);
            const uint32_t offset = rowOffset + col;
            LoadGateRegbasePair<VType>(vZeroReg, vOneReg, vIn + offset, inputMask, inputReg);
            Mul(vZeroReg, vZeroReg, betaReg, floatMask);
            Mul(vOneReg, vOneReg, betaReg, floatMask);
            if (row >= validRows) {
                Duplicate(vZeroReg, 0.0f, floatMask);
                Duplicate(vOneReg, 0.0f, floatMask);
            }
            CastFloat2Half<VType>(outReg, vZeroReg, vOneReg, floatMask);
            StoreAlign(vbOut + offset, outReg, inputMask);
        }
    }
}

} // namespace KdaBwdRecomputeArch35

#endif // __CCE_AICORE__ == 310

#endif // CHUNK_KDA_BWD_RECOMPUTE_ARCH35_REGBASE_H
