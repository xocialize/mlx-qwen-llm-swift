import Foundation
import MLXToolKit

/// Init-time configuration for `QwenLLMPackage` (C9). Carries the chosen checkpoint
/// (size × quant) and an optional pinned revision; everything that changes call-to-call
/// (prompt, sampling, mode) rides the `LLMRequest`, never here.
public struct QwenLLMConfiguration: PackageConfiguration, ModelStorable {
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

    // `modelsRootDirectory` is intentionally excluded — environment-specific, not portable config.
    private enum CodingKeys: String, CodingKey {
        case model, revision, memoryBudgetBytes
    }
}
