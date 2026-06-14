# mlx-qwen-llm-swift

An [MLXEngine](https://github.com/xocialize/mlx-engine-swift) model package exposing the **`llm`**
capability over Qwen3.5 (default: 0.8B, 8-bit) on Apple silicon via
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

It conforms to the `ModelPackage` contract in `MLXToolKit`: a `PackageManifest` (capabilities,
requirements per quant, license), lazy `load()`, and a `run()` that maps the canonical
`LLMRequest`/`LLMResponse` to an MLX `ChatSession` with multi-turn history. The
`MLXServeEngine` coordinator handles licensing, device eligibility, and memory budgeting.

## Models

`QwenModel.allPublished` catalogs the supported Qwen3.5 sizes × quants; consumers select one
through `QwenLLMConfiguration`. Weights download on first use from the configured Hugging Face repo.

## Usage

```swift
import MLXServeCore
import MLXQwenLLM

let engine = MLXServeEngine()
try await engine.register(QwenLLMPackage.registration, configuration: QwenLLMConfiguration())
try await engine.prepare(.llm)
let response = try await engine.run(LLMRequest(messages: [.init(role: .user, content: "Hi")]))
```

## Development

This package is co-developed inside the MLXEngine workspace and consumes the engine as a tagged-URL
net dependency (`.package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.3.0")`), so
it builds standalone without a local checkout.

## License

MIT — the Swift port. Qwen3.5 weights are licensed by their publisher (Apache-2.0); review the
model card before redistribution.
