// swift-tools-version: 6.2
import PackageDescription

// mlx-qwen-llm-swift — the first MLXEngine package: a Qwen3.5 `llm` surface backed by the
// MLX-Swift LM runtime. It conforms to the MLXEngine contract (MLXToolKit) and is consumed by
// the engine via SPM. Kept separate from `mlx-engine-swift` so the contract package stays free
// of the heavy MLX dependency.
//
// Naming: Swift ports carry the `-swift` suffix on the package/repo name (mirroring
// `mlx-engine-swift`) to differentiate from the Python ports; the module/product stays clean
// PascalCase (`MLXQwenLLM`).
//
// PHASE A (current): contract-side only — depends on MLXToolKit, so it builds offline against the
// local engine package and proves the ModelPackage conformance. Inference is stubbed.
//
// PHASE B: uncomment the mlx-swift-lm dependency below and implement load()/run() in
// QwenLLMPackage against MLXLLM / MLXLMCommon (ChatSession / LLMModelFactory).
let package = Package(
    name: "mlx-qwen-llm-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXQwenLLM", targets: ["MLXQwenLLM"]),
    ],
    dependencies: [
        // ≥0.45.0: contract 1.33.0 `LLMParameters.seed` — the canonical RNG pin this package
        // forwards to GenerateParameters (AB-R-0079). On top of ≥0.27.0's run-lifecycle V4 / CAN
        // cancellation-conformance gate (MLXServeConformance CAN-1..3) and contract 1.16.0
        // responseFormat.
        // ≥0.82.1: Quant.mxfp8/.nvfp4 (contract 1.60.0, engine 0.75.0) are named in QwenModel's
        // exhaustive switches, and mlx-swift comes from the fleet's 0.33.1 fork (AB-D-0264).
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.82.1"),
        // JSON grammar-constrained decoding (the responseFormat runtime, shared with the
        // gemma package).
        .package(url: "https://github.com/xocialize/mlx-constrained-decoding-swift", from: "0.1.0"),
        // MLX-Swift LM runtime (https://github.com/ml-explore/mlx-swift-lm):
        // 3.33.0 is the first release on mlx-swift 0.33, which the fork below needs.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.33.0")),
        // mlx-swift from the fleet's fork: 0.33.1 = upstream 0.33.0 with the vendored MLX's M5
        // head-dim-split attention routes disabled (AB-R-1009, AB-D-0264). Upstream 0.33.0 aborts
        // on M5 for head-256 masked prefills of 1024+ tokens (Qwen 3.5 class models, head dim 256, hit it). This line is the ROOT
        // declaration when the package builds on its own, so it has to name the fork; same
        // package identity as ml-explore's. Back to upstream once a release gates clean.
        .package(url: "https://github.com/xocialize/mlx-swift", from: "0.33.1"),
        // mlx-swift-lm 3.x decoupled the HF stack — the download macro/tokenizer needs these
        // provided by the consumer. Versions pinned to a known-good resolved set.
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.2.1"),
    ],
    targets: [
        .target(
            name: "MLXQwenLLM",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXConstrainedDecoding", package: "mlx-constrained-decoding-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                // Both factories linked so loadModelContainer auto-dispatches Qwen3.5 by
                // config.json (text variant lives in MLXLLM, vision variant in MLXVLM).
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                // HF download + tokenizer for the #huggingFaceLoadModelContainer macro.
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // Live gates (GPU inference runs here, not in the SPM test product, whose metallib is
        // unreliable — the fleet CLI-gate convention): --smoke / --kv-reuse / --mem-bench.
        .executableTarget(
            name: "RunQwenLLM",
            dependencies: [
                "MLXQwenLLM",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXQwenLLMTests",
            dependencies: [
                "MLXQwenLLM",
                // Test-only: run the variant catalog through the engine's admissibility check.
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                // Test-only: the offline CAN cancellation-conformance gate (CAN-1..3).
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
