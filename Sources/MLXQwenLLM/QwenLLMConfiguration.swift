import Foundation
import MLXToolKit

/// Init-time configuration for `QwenLLMPackage` (C9). Carries the chosen checkpoint
/// (size × quant) and an optional pinned revision; everything that changes call-to-call
/// (prompt, sampling, mode) rides the `LLMRequest`, never here.
///
/// **BudgetAware — the intended future LLM lever (deferred).** Unlike a diffusion package whose
/// memory lever is the quant chosen at config, an LLM has a load-time lever the engine could drive:
/// **cap the context (max total tokens) so the KV-cache fits the admitted budget** — a smaller
/// context is a smaller transient (see `QwenModel.kvCacheBytes`). Adopting `BudgetAware` here would
/// let `load()` derive a safe context from `availableBudgetBytes − weights` and clamp generation to
/// it. We DEFER this: the KV-cache at the documented envelope (≤256 MB even at 9B) is small relative
/// to the weights, so it rarely decides admission today, and silently shrinking a caller's context
/// is a correctness surprise we don't want without a constrained tier that needs it. Documented as
/// the future lever; not implemented.
public struct QwenLLMConfiguration: PackageConfiguration, ModelStorable, FootprintConfigured {
    /// Which Qwen3.5 checkpoint to materialize and load.
    public var model: QwenModel
    /// Pinned weights revision (commit/tag). `nil` resolves to the repo default.
    public var revision: String?
    /// Optional resident-memory budget hint for the engine's MemoryPool placement.
    public var memoryBudgetBytes: UInt64?
    /// Where weights are cached/materialized. When set, the engine-chosen models folder is used
    /// (the caller must hold security-scoped access). When `nil`, the default HubApi cache is used.
    /// Excluded from `Codable` — a URL is environment-specific, not part of the portable config.
    public var modelsRootDirectory: URL?

    public init(model: QwenModel = .default,
                revision: String? = nil,
                memoryBudgetBytes: UInt64? = nil,
                modelsRootDirectory: URL? = nil) {
        self.model = model
        self.revision = revision
        self.memoryBudgetBytes = memoryBudgetBytes
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// The HF `mlx-community` repo id for the chosen checkpoint, if published.
    public var weightsRepo: String? { model.weightsRepo }

    // MARK: FootprintConfigured — the selected (size × quant) variant's split footprint
    //
    // The footprint varies along TWO axes — size *and* quant — so a quant-keyed `QuantFootprint`
    // (and `QuantConfigured`) can't express it: 0.8B-bf16 and 4B-bf16 are the same quant but very
    // different working sets. This is the BiRefNet-style per-config-hint case. The hints declare the
    // chosen checkpoint's exact split so the governor charges it precisely instead of the static
    // manifest's default-variant figure.

    /// Persistent weights floor of the selected checkpoint.
    public var residentBytesHint: UInt64? { model.residentBytes }

    /// Transient KV-cache (+ scratch) of the selected checkpoint at the documented context envelope.
    public var peakActivationBytesHint: UInt64? { model.peakActivationBytes }

    // `modelsRootDirectory` is intentionally excluded — environment-specific, not portable config.
    private enum CodingKeys: String, CodingKey {
        case model, revision, memoryBudgetBytes
    }
}

// MARK: - Weight sources (the gemma package's MAT pattern)

extension QwenLLMConfiguration: WeightSourcing {
    /// One source: the configured checkpoint's repo. Without it the engine stamped its install
    /// marker on the static manifest's `provenance.sourceRepo` (the 0.8B default) whatever
    /// checkpoint loaded, and `needsDownload` read that same marker (AB-A-0195). Declared, the
    /// engine marks the repo that actually loaded and `needsDownload` probes it (MS-6). The quant
    /// is tiered by repo (`Qwen3.5-<size>-MLX-<quant>`), so `matching` stays nil: the whole
    /// snapshot. Unpublished quants (`weightsRepo == nil`) declare nothing; `load()` rejects them.
    ///
    /// `missingWeightSources` is the engine's default probe (MS-2): the hub snapshot
    /// (`models--<org>--<name>/snapshots/<commit>/`, what this package's own download writes) or
    /// the engine-executed flat layout (`models--<org>--<name>/`, contract 1.24).
    public var weightSources: [WeightSource] {
        guard let repo = weightsRepo else { return [] }
        return [WeightSource(role: "checkpoint", repo: repo, revision: revision)]
    }

    /// Where `load()` reads a materialized checkpoint from under `storeRoot`, judged by the same
    /// probe as `missingWeightSources`: the hub snapshot when it satisfies the source, else the
    /// flat repo directory when that does. `nil` when neither does (or there's no store), so
    /// `load()` downloads it.
    public func resolvedModelDirectory(storeRoot: URL?) -> URL? {
        guard let repo = weightsRepo else { return nil }
        let store = ModelStore(root: storeRoot)
        if let snapshot = store.snapshotDirectory(for: repo, revision: revision),
           WeightSourceProbe.snapshot(snapshot, satisfies: nil) {
            return snapshot
        }
        if let flat = store.directory(for: repo), WeightSourceProbe.flatDirectory(flat, satisfies: nil) {
            return flat
        }
        return nil
    }
}
