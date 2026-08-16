# mlx-qwen-llm-swift

An [MLXEngine](https://github.com/xocialize/mlx-engine-swift) model package exposing the **`llm`**
capability over Qwen3.5 (default: 0.8B, 8-bit) on Apple silicon via
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

It conforms to the `ModelPackage` contract in `MLXToolKit`: a `PackageManifest` (capabilities,
requirements per quant, license), lazy `load()`, and a `run()` that maps the canonical
`LLMRequest`/`LLMResponse` to an MLX `ChatSession` with multi-turn history. The
`MLXServeEngine` coordinator handles licensing, device eligibility, and memory budgeting.

The package holds one `ChatSession` across `run()` calls (**KV-cache reuse**): when a request
is exactly the previous transcript plus one new user turn, only that turn is prefilled —
per-turn latency and the prefill transient stay flat as a conversation grows. Any transcript
mismatch falls back to a fresh session. The retained KV cache is intentional active-memory
retention, dropped on `unload()`; hit/miss counts are exposed (`kvReuseHits`/`kvReuseMisses`)
and logged (`Logger` subsystem `MLXQwenLLM`, category `kv-reuse`).

Sampling is pinnable: set `parameters.seed` on the request (contract 1.33.0, engine ≥ 0.45.0) and
the same `(prompt, seed)` reproduces; `nil` — the default — seeds from system entropy, and the
field is inert at `temperature == 0`. One caveat from the KV reuse above: the guarantee is
per-**call**, not per-conversation. A pinned one-shot call reproduces outright; pinning turn 5 of
a held conversation reproduces only after replaying turns 1–4, because the session state is part
of the input.

## Live gates

GPU gates run via the CLI (fleet convention — not in the SPM test product):

```
swift run -c release RunQwenLLM --smoke        # one governed-shape generate
swift run -c release RunQwenLLM --kv-reuse     # reuse correctness A/B + latency + fallback
swift run -c release RunQwenLLM --mem-bench    # split-footprint + held-KV retention drift
```

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
