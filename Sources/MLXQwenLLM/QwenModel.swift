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
}

extension Quant {
    /// The mlx-community repo suffix for this quant when one is published
    /// (`4bit` / `8bit` / `bf16`); `nil` for quants mlx-community doesn't ship under that scheme.
    public var mlxCommunitySuffix: String? {
        switch self {
        case .int4: return "4bit"
        case .int8: return "8bit"
        case .bf16: return "bf16"
        case .fp16, .mxfp4: return nil
        }
    }

    /// Approximate bytes-per-weight, for on-disk / resident footprint estimation.
    fileprivate var bytesPerWeight: Double {
        switch self {
        case .int4, .mxfp4: return 0.5
        case .int8:         return 1.0
        case .bf16, .fp16:  return 2.0
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

    /// Approximate resident footprint (weights + runtime/cache headroom), in bytes.
    public var residentBytes: UInt64 {
        onDiskBytes + 600_000_000
    }

    /// Cost-to-run footprint for the Model Manager (C10).
    public var footprint: QuantFootprint {
        QuantFootprint(quant: quant, residentBytes: residentBytes)
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
