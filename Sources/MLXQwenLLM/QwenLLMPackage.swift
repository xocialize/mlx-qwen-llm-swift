import Foundation
import MLXToolKit
import MLXLMCommon
import MLXLLM
import MLXVLM
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// The first MLXEngine package: a Qwen3.5 (4-bit) model exposing the canonical `llm` surface.
///
/// One `ModelPackage`, one surface. The engine owns the lifecycle (inversion of control): it
/// constructs this from a `QwenLLMConfiguration`, pages weights in with `load()`, drives
/// `run(_:)`, and reclaims with `unload()`. Lifecycle methods are isolated to `InferenceActor`
/// (the class is annotated), so C13 ("runs only in the serialization domain, no private queue")
/// is compiler-enforced and no internal locking is needed.
///
/// PHASE A: inference is stubbed (see the `TODO(Phase B)` markers). The conformance, manifest,
/// and registration are real. PHASE B wires `load()`/`run()` to the MLX-Swift LM runtime.
@InferenceActor
public final class QwenLLMPackage: ModelPackage {
    public typealias Configuration = QwenLLMConfiguration

    /// Static, registrable blueprint — read at registration/eligibility time, before any
    /// instance exists, so it is `nonisolated`. It describes the default variant (0.8B); the
    /// per-size footprint that differs is selected at construction via `Configuration`.
    public nonisolated static var manifest: PackageManifest {
        let target = QwenModel.default
        return PackageManifest(
            // Qwen3.5 weights are Apache-2.0; this port code is Apache-2.0 too. Both layers permissive.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            // TODO(Phase B): pin `revision` to the exact published commit (provenance-lint).
            provenance: Provenance(
                sourceRepo: target.weightsRepo ?? "mlx-community/Qwen3.5-0.8B-MLX-8bit",
                revision: "main",
                tier: 1
            ),
            // Single source of truth for C10 requirements (footprint/backends/OS).
            requirements: target.requirements,
            specialties: [
                SpecialtyWeight(.general, strength: 0.8),
                SpecialtyWeight(.coder, strength: 0.6),
            ],
            surfaces: [
                LLMContract.descriptor(
                    name: "qwen3.5-llm",
                    summary: "Qwen3.5 text generation (MLX, 4-bit).",
                    modes: [.direct, .thinking]
                )
            ]
        )
    }

    private let configuration: Configuration
    /// The resident model + tokenizer, paged in by `load()`. `nil` until loaded.
    private var container: ModelContainer?

    /// Cheap construction — no compute, no weight paging (C13). Residency is `load()`'s job.
    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Page the working set in. Idempotent when already resident.
    ///
    /// Loads via mlx-swift-lm's HF loader, which downloads + caches the weights and dispatches
    /// to the matching factory (MLXLLM or MLXVLM) by the model's `config.json`. Phase 3 will
    /// redirect the download into the engine-chosen models folder.
    public func load() async throws {
        guard container == nil else { return }
        guard let repo = configuration.weightsRepo else {
            throw PackageError.configurationMismatch(
                expected: "a published mlx-community quant",
                got: configuration.model.displayName)
        }
        let revision = configuration.revision ?? "main"
        if let root = configuration.modelsRootDirectory {
            // Materialize into the engine-chosen models folder (caller holds security-scoped
            // access) by pointing the Hub cache there, instead of the default container cache.
            let client = HubClient(cache: HubCache(cacheDirectory: root))
            container = try await loadModelContainer(
                from: #hubDownloader(client),
                using: #huggingFaceTokenizerLoader(),
                id: repo,
                revision: revision)
        } else {
            let modelConfig = ModelConfiguration(id: repo, revision: revision)
            container = try await #huggingFaceLoadModelContainer(configuration: modelConfig)
        }
    }

    /// Release the working set; the instance survives for a later `load()`.
    public func unload() async {
        container = nil
    }

    /// Run one `llm` call. Dispatches on capability, decodes the canonical request, generates
    /// text on the resident model, and returns canonical text. Honors cancellation so the
    /// MemoryGovernor can preempt + requeue.
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let container else { throw PackageError.notLoaded }
        guard request.capability == .llm, let llm = request as? LLMRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        // Map canonical sampling controls onto MLX GenerateParameters.
        var parameters = GenerateParameters()
        if let temperature = llm.parameters.temperature { parameters.temperature = Float(temperature) }
        if let topP = llm.parameters.topP { parameters.topP = Float(topP) }
        parameters.maxTokens = llm.parameters.maxTokens

        // System turns become session instructions.
        let instructions = llm.messages
            .filter { $0.role == .system }
            .map(\.content)
            .joined(separator: "\n\n")

        // Multi-turn: seed every prior non-system turn as history, respond to the final user turn.
        // A single message yields empty history (same as one-shot).
        let conversational = llm.messages.filter { $0.role != .system }
        let prompt = conversational.last?.content ?? ""
        let priorTurns = conversational.isEmpty ? [] : Array(conversational.dropLast())

        let text = try await Self.generate(
            container: container,
            instructions: instructions.isEmpty ? nil : instructions,
            history: priorTurns,
            prompt: prompt,
            parameters: parameters
        )
        return LLMResponse(text: text, finishReason: .stop)
    }

    /// Runs one generation on a fresh `ChatSession`. `ChatSession` is not `Sendable` (and not
    /// thread-safe), so it is created and consumed entirely inside this `nonisolated` helper from
    /// `Sendable` inputs — it never crosses the `InferenceActor` isolation boundary (which would
    /// otherwise trip the "sending risks data races" check). `ModelContainer` provides its own
    /// internal isolation.
    private nonisolated static func generate(
        container: ModelContainer,
        instructions: String?,
        history: [ChatMessage],
        prompt: String,
        parameters: GenerateParameters
    ) async throws -> String {
        let chatHistory: [Chat.Message] = history.map { message in
            switch message.role {
            case .assistant: return .assistant(message.content)
            case .user, .system: return .user(message.content)
            }
        }
        let session = ChatSession(
            container,
            instructions: instructions,
            history: chatHistory,
            generateParameters: parameters
        )
        return try await session.respond(to: prompt)
    }
}

extension QwenLLMPackage {
    /// The author one-liner the engine registers: manifest + license-gated factory. Read at
    /// registration time (outside the serialization domain), so `nonisolated`.
    public nonisolated static var registration: PackageRegistration {
        .of(QwenLLMPackage.self)
    }
}
