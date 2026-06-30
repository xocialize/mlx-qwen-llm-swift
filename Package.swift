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
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.15.0"),
        // MLX-Swift LM runtime (https://github.com/ml-explore/mlx-swift-lm):
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
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
        .testTarget(
            name: "MLXQwenLLMTests",
            dependencies: [
                "MLXQwenLLM",
                // Test-only: run the variant catalog through the engine's admissibility check.
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
