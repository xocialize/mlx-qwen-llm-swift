import Foundation
import MLXToolKit

/// A Qwen3.5 model size. Repo ids follow the mlx-community convention
/// `Qwen3.5-<size>-MLX-<quant>`.
public enum QwenSize: String, Sendable, Codable, CaseIterable {
    case b0_8 = "0.8B"
    case b4   = "4B"
    case b9   = "9B"

    /// Approximate parameter count in billions — used for footprint estimates.
    public var paramsBillions: Double {
        switch self {
        case .b0_8: return 0.8
        case .b4:   return 4
        case .b9:   return 9
        }
    }

    // MARK: KV-cache geometry (from the published `config.json` of each size)
    //
    // Qwen3.5 is a HYBRID linear/full-attention transformer: only every `fullAttentionInterval`-th
    // layer is full softmax attention with a context-growing `KVCacheSimple`; the others are
    // GatedDeltaNet (linear) layers whose `MambaCache` is a FIXED-SIZE recurrent state that does NOT
    // grow with context. So the autoregressive KV-cache transient is driven by the full-attention
    // layers ALONE — counting all layers (as a vanilla-transformer formula would) over-states it ~4×.
    // Values verified against mlx-community/Qwen3.5-{0.8B,4B,9B}-MLX-* config.json (2026-06-30).

    /// Total decoder layers (`num_hidden_layers`).
    var numLayers: Int {
        switch self {
        case .b0_8: return 24
        case .b4:   return 32
        case .b9:   return 32
        }
    }

    /// KV heads on the full-attention layers (`num_key_value_heads`, GQA).
    var numKVHeads: Int {
        switch self {
        case .b0_8: return 2
        case .b4:   return 4
        case .b9:   return 4
        }
    }

    /// Per-head dimension on the full-attention layers (`head_dim`, 256 across the family).
    var headDim: Int { 256 }

    /// Model width (`hidden_size`) — the scale factor for the prefill-scratch activation peak.
    var hiddenSize: Int {
        switch self {
        case .b0_8: return 1024
        case .b4:   return 2560
        case .b9:   return 4096
        }
    }

    /// 1-in-`fullAttentionInterval` layers is full attention; the rest are linear. (`full_attention_interval`.)
    var fullAttentionInterval: Int { 4 }

    /// Count of full-attention (KV-cached) layers: layers where `(idx+1) % interval == 0`.
    /// = `numLayers / fullAttentionInterval` (24/4=6, 32/4=8).
    var numFullAttentionLayers: Int { numLayers / fullAttentionInterval }
}

extension Quant {
    /// The mlx-community repo suffix for this quant when one is published
    /// (`4bit` / `8bit` / `bf16`); `nil` for quants mlx-community doesn't ship under that scheme.
    public var mlxCommunitySuffix: String? {
        switch self {
        case .int4: return "4bit"
        case .int8: return "8bit"
        case .bf16: return "bf16"
        case .fp16, .fp32, .mxfp4, .int5, .int6: return nil
        }
    }

    /// Approximate bytes-per-weight, for on-disk / resident footprint estimation.
    fileprivate var bytesPerWeight: Double {
        switch self {
        case .int4, .mxfp4: return 0.5
        case .int5:         return 0.625
        case .int6:         return 0.75
        case .int8:         return 1.0
        case .bf16, .fp16:  return 2.0
        case .fp32:         return 4.0
        }
    }
}

/// A concrete Qwen3.5 checkpoint = size × quantization. Confirmed-published combinations on
/// mlx-community: 0.8B and 4B in {4bit, 8bit, bf16}, 9B in {4bit}.
public struct QwenModel: Sendable, Codable, Equatable, Hashable {
    public var size: QwenSize
    public var quant: Quant

    public init(size: QwenSize, quant: Quant) {
        self.size = size
        self.quant = quant
    }

    /// First-bring-up default: **0.8B 8-bit** (confirmed published, ~1 GB on disk).
    public static let `default` = QwenModel(size: .b0_8, quant: .int8)

    public var displayName: String { "Qwen3.5 · \(size.rawValue) (\(quant.rawValue))" }

    /// HF `mlx-community` repo id, e.g. `mlx-community/Qwen3.5-0.8B-MLX-8bit`. `nil` when the
    /// quant has no published mlx-community suffix.
    public var weightsRepo: String? {
        guard let suffix = quant.mlxCommunitySuffix else { return nil }
        return "mlx-community/Qwen3.5-\(size.rawValue)-MLX-\(suffix)"
    }

    /// Approximate on-disk size of the materialized weights, in bytes.
    public var onDiskBytes: UInt64 {
        UInt64(size.paramsBillions * 1_000_000_000 * quant.bytesPerWeight)
    }

    /// The persistent weights floor — what stays resident the whole time the model is loaded.
    /// This is the on-disk weight bytes of the *selected* checkpoint (mmap'd, paged in on demand).
    /// The autoregressive transient (the KV-cache) is split out into `peakActivationBytes`, NOT
    /// folded in here, so the engine can reserve a single shared activation across co-residents.
    public var residentBytes: UInt64 { onDiskBytes }

    /// The documented max-context envelope the declared footprint is sized for: prompt + generated
    /// tokens. Both the (small) persisted KV-cache and the (dominant) prefill activation scratch scale
    /// ~linearly with this. Chosen as a generous chat working window (8192) — far below
    /// `max_position_embeddings` (131072), the analog of the diffusion packages' documented resolution
    /// envelope. A request that runs past this still works; it just exceeds the declared transient (the
    /// reactive `phys_footprint` governor trigger still catches a true OOM). 8192 was rejected: the
    /// prefill scratch there is ~7 GB even on 0.8B (see `peakActivationBytes`), pathological for a chat
    /// surface; 2048 is the realistic working window and keeps the declared reserve proportionate.
    public static let contextEnvelopeTokens = 2048

    /// The autoregressive KV-cache size, in bytes, for a given total context (`maxTokens` = prompt +
    /// generated). This is THE transient lever for an LLM — unlike a diffusion DiT's fixed activation
    /// peak, the cache grows linearly with context.
    ///
    /// Formula: `2 (K+V) × fullAttentionLayers × kvHeads × headDim × maxTokens × cacheDtypeBytes`.
    /// - **Full-attention layers only** — the GatedDeltaNet (linear) layers use a fixed-size
    ///   `MambaCache` that does not grow with context, so they contribute O(1), not O(context).
    /// - **KV heads** (GQA), not query heads.
    /// - The cache is stored at the model's compute dtype (bf16, 2 B/elem) regardless of weight quant
    ///   — quantization shrinks the weights, not the attention cache.
    public func kvCacheBytes(maxTokens: Int) -> UInt64 {
        let cacheDtypeBytes = 2          // bf16 K/V cache
        let kPlusV = 2
        let elements = kPlusV
            * size.numFullAttentionLayers
            * size.numKVHeads
            * size.headDim
            * max(0, maxTokens)
        return UInt64(elements * cacheDtypeBytes)
    }

    /// The transient activation peak at the documented context envelope.
    ///
    /// **Measurement finding (2026-06-30, the interesting bit):** the analytic *persisted* KV-cache
    /// (`kvCacheBytes`, verified bit-exact at 12 288 B/token for 0.8B) is NOT the activation peak for
    /// this hybrid architecture — it is two orders of magnitude too small. The real transient is
    /// dominated by **prefill compute scratch** that scales ~linearly with the prompt sequence length:
    /// the GatedDeltaNet (linear-attention) chunked-scan intermediates over the prompt, not the softmax
    /// KV-cache. Measured on 0.8B-8bit via the engine's defaults (`prefillStepSize` 512):
    ///
    ///   prompt ~340 tok → 427 MB · ~1k → 1.6 GB · **~2k (envelope) → ~2.1 GB** · ~4k → 4.2 GB · ~8k → ~7.0 GB
    ///
    /// So the peak is the prefill scratch at the envelope, ≈ `1.03 MB × contextEnvelopeTokens` for the
    /// 0.8B width, scaled by `hidden_size` for the larger sizes (only 0.8B is measured; 4B/9B are the
    /// width-scaled estimate, marked for re-measure when those variants are validated). This is exactly
    /// the "flat footprints can UNDER-declare" lesson — declaring the analytic KV-cache alone (96 MB
    /// @8192) would have under-reserved by ~20×. The analytic `kvCacheBytes` is kept (it is the correct
    /// *persisted* cache and the basis for the future BudgetAware context-cap lever), but the declared
    /// peak is empirical.
    public var peakActivationBytes: UInt64 {
        // Measured 0.8B prefill-scratch at the 2048-token envelope (~2.1 GB ⇒ ~1.03 MB/token).
        let bytesPerTokenAt0_8B = 1_075_000.0
        let widthScale = Double(size.hiddenSize) / Double(QwenSize.b0_8.hiddenSize)
        let prefillScratch = bytesPerTokenAt0_8B * Double(Self.contextEnvelopeTokens) * widthScale
        // The persisted KV-cache rides on top of (is subsumed by, but add for safety) the scratch.
        return UInt64(prefillScratch) + kvCacheBytes(maxTokens: Self.contextEnvelopeTokens)
    }

    /// Cost-to-run footprint for the Model Manager (C10): the persistent weights floor plus the
    /// split-out transient activation reserve (measured prefill-scratch at the documented envelope).
    public var footprint: QuantFootprint {
        QuantFootprint(quant: quant,
                       residentBytes: residentBytes,
                       peakActivationBytes: peakActivationBytes)
    }

    /// The C10 requirements for this exact checkpoint — what the engine's `DeviceProfile` +
    /// `MemoryGovernor` evaluate to decide if the machine can load it. Memory (the footprint vs the
    /// budget) is the real capability gate here; all variants run on the Metal GPU and need macOS 26.
    public var requirements: RequirementsManifest {
        RequirementsManifest(
            footprints: [footprint],
            requiredBackends: [.metalGPU],
            os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
            chipFloor: nil
        )
    }

    /// Every checkpoint published on mlx-community (the catalog). 0.8B and 4B ship in
    /// {4-bit, 8-bit, bf16}; 9B ships in 4-bit. These double as sanity markers: run each through the
    /// engine's `admissibility(for:)` to see what the current machine can actually load.
    public static let allPublished: [QwenModel] = [
        QwenModel(size: .b0_8, quant: .int4),
        QwenModel(size: .b0_8, quant: .int8),
        QwenModel(size: .b0_8, quant: .bf16),
        QwenModel(size: .b4, quant: .int4),
        QwenModel(size: .b4, quant: .int8),
        QwenModel(size: .b4, quant: .bf16),
        QwenModel(size: .b9, quant: .int4),
    ]
}
