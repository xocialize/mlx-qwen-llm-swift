import Foundation
import MLX
import MLXToolKit
import MLXConstrainedDecoding
import MLXLMCommon
import MLXLLM
import MLXVLM
import MLXHuggingFace
import HuggingFace
import Tokenizers
import os

/// The first MLXEngine package: a Qwen3.5 model (default 0.8B, 8-bit) exposing the canonical
/// `llm` surface.
///
/// One `ModelPackage`, one surface. The engine owns the lifecycle (inversion of control): it
/// constructs this from a `QwenLLMConfiguration`, pages weights in with `load()`, drives
/// `run(_:)`, and reclaims with `unload()`. Lifecycle methods are isolated to `InferenceActor`
/// (the class is annotated), so C13 ("runs only in the serialization domain, no private queue")
/// is compiler-enforced and no internal locking is needed.
///
/// `load()`/`run()` are wired to the MLX-Swift LM runtime: `load()` pages weights via the HF
/// loader, `run(_:)` maps the canonical request onto a `ChatSession` and returns canonical text.
///
/// **KV-cache reuse (prompt caching).** The package holds ONE `ChatSession` across `run(_:)`
/// calls. When an incoming request is exactly the held session's transcript plus one new user
/// turn (see `SessionReuse.decision`), only that turn is sent — the session's retained
/// `[KVCache]` already encodes the prefix, so per-turn latency and the prefill-scratch
/// transient stay flat instead of growing with conversation length (measured 2.1 GB @2k →
/// 7 GB @8k prefill scratch on 0.8B-8bit; GatedDeltaNet chunked-scan). Any mismatch rebuilds a
/// fresh session (the pre-reuse behavior). **Memory triage note:** the retained KV cache (plus
/// the GatedDeltaNet recurrent state) is *intentional* active-memory retention that grows with
/// the conversation — `gpuPoolSnapshot` "active climbing across turns" while this package is
/// chatting is expected, not a leak; it is dropped on `unload()` (eviction) and on any
/// transcript mismatch.
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
                    summary: "Qwen3.5 text generation (MLX).",
                    modes: [.direct, .thinking],
                    // Honest C11 advertisement: this package really constrains
                    // (grammar-masked decode via MLXConstrainedDecoding, contract 1.16.0).
                    supportsStructuredOutput: true
                )
            ]
        )
    }

    private let configuration: Configuration
    /// The resident model + tokenizer, paged in by `load()`. `nil` until loaded.
    private var container: ModelContainer?

    /// The held multi-turn session whose KV cache encodes `held.transcript`. `nil` until the
    /// first eligible run; dropped on `unload()`, on any transcript mismatch, and on any error
    /// mid-generation (a throw can leave the KV cache with a partially appended turn the
    /// transcript doesn't record — reusing it would be a silent-corruption false positive).
    private var held: HeldChatSession?

    /// Escape hatch + A/B seam: `false` forces a fresh session per run (the pre-reuse
    /// behavior). Used by the `RunQwenLLM --kv-reuse` gate to produce the baseline leg.
    public var kvCacheReuseEnabled = true

    /// Vocab classification for constrained decoding (contract 1.16.0 `responseFormat`) —
    /// built once per residency on the first structured request (a full-vocab pass over the
    /// tokenizer; not paid by freeform-only consumers), dropped with the weights on `unload()`.
    private var constrainedVocabulary: TokenVocabulary?

    /// Masking telemetry of the most recent structured run (steps, wall-clock spent in the
    /// per-step allowed-set computation, simulated candidates) — surfaced for the
    /// `RunQwenLLM --structured` gate's latency report.
    public private(set) var lastStructuredStats: JSONConstraintEngine.Stats?

    /// KV-reuse observability: cumulative counts since load. Surfaced so a consuming app can
    /// verify its hit rate in situ; each run also logs a HIT/MISS line via `Logger`
    /// (subsystem `MLXQwenLLM`, category `kv-reuse`).
    public private(set) var kvReuseHits = 0
    public private(set) var kvReuseMisses = 0

    private nonisolated static let log = Logger(subsystem: "MLXQwenLLM", category: "kv-reuse")

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
        held = nil                // the retained KV cache goes with the weights (its intended lifetime)
        constrainedVocabulary = nil
        container = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    /// Run one `llm` call. Dispatches on capability, decodes the canonical request, generates
    /// text on the resident model, and returns canonical text. Honors cancellation so the
    /// MemoryGovernor can preempt + requeue.
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: both generation paths bail per generated token —
        // mlx-swift-lm's generation loop checks Task.isCancelled every token (plain path;
        // MLXLMCommon/Evaluate.swift tokenLoop, 3.31.4) with the post-respond checkpoint in
        // `respond(in:to:...)` rethrowing, and the structured TokenIterator drive checks per
        // token directly.
        try Task.checkCancellation()
        guard let container else { throw PackageError.notLoaded }
        guard request.capability == .llm, let llm = request as? LLMRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        // Map canonical sampling controls onto MLX GenerateParameters.
        var parameters = GenerateParameters()
        if let temperature = llm.parameters.temperature { parameters.temperature = Float(temperature) }
        if let topP = llm.parameters.topP { parameters.topP = Float(topP) }
        parameters.maxTokens = llm.parameters.maxTokens

        // Mode → chat-template kwargs. Additive: a `nil` mode injects nothing, so existing
        // callers are byte-for-byte unchanged; only an explicit `.direct`/`.companion`/`.thinking`
        // sets `enable_thinking` on the Qwen3.5 hybrid-reasoner template.
        let templateContext = Self.templateContext(for: llm.mode)

        // Structured output (contract 1.16.0): bypass ChatSession entirely — mlx-swift-lm
        // 3.31.x has no processor-injection seam through GenerateParameters/ChatSession, so
        // the request is templated directly and driven through TokenIterator with the
        // grammar-masking LogitProcessor and a FRESH cache. Deliberately does not touch
        // `held`: the companion's structured calls are one-shot side calls, and the
        // conversational KV-reuse path stays byte-for-byte unchanged.
        if llm.responseFormat != nil {
            return try await runStructured(container: container, request: llm,
                                           parameters: parameters,
                                           templateContext: templateContext)
        }

        let incoming = SessionTranscript(messages: llm.messages)
        let decision = kvCacheReuseEnabled
            ? SessionReuse.decision(held: held?.transcript, incoming: incoming)
            : .rebuild

        let text: String
        if case .reuse(let newUserTurn) = decision, let session = held {
            // Cache hit: the held KV cache encodes everything but this turn, so send ONLY the
            // new turn (TokenIterator appends the full templated input at the cache offset —
            // no prefix dedupe). Take the box out first: if respond throws, `held` stays nil
            // and the next run rebuilds instead of reusing a partially-appended cache.
            held = nil
            kvReuseHits += 1
            Self.log.debug("HIT — appending 1 turn at cache offset (hits=\(self.kvReuseHits) misses=\(self.kvReuseMisses))")
            text = try await Self.respond(in: session,
                                          to: newUserTurn,
                                          parameters: parameters,
                                          additionalContext: templateContext)
            session.transcript.turns.append(ChatMessage(role: .user, content: newUserTurn))
            session.transcript.turns.append(ChatMessage(role: .assistant, content: text))
            held = session
        } else {
            // Fresh session (first turn, reuse disabled, or transcript mismatch): seed every
            // prior non-system turn as history and respond to the final turn — the pre-reuse
            // behavior, byte-identical templating. A single message yields empty history.
            held = nil
            kvReuseMisses += 1
            Self.log.debug("MISS — fresh session, prefilling \(incoming.turns.count) turn(s) (hits=\(self.kvReuseHits) misses=\(self.kvReuseMisses))")
            let prompt = incoming.turns.last?.content ?? ""
            let priorTurns = incoming.turns.isEmpty ? [] : Array(incoming.turns.dropLast())
            let session = Self.startSession(container: container,
                                            systemPrompt: incoming.systemPrompt,
                                            priorTurns: priorTurns,
                                            parameters: parameters,
                                            additionalContext: templateContext)
            text = try await Self.respond(in: session,
                                          to: prompt,
                                          parameters: parameters,
                                          additionalContext: templateContext)
            // Hold the session for reuse only when the turn we just answered really was a user
            // turn — a trailing assistant/degenerate turn was *templated as user* in this
            // session's KV, so a later fresh rebuild (which templates it by its true role)
            // would diverge from the cache. Not holding keeps those callers on today's path.
            // The responded turn is dropLast'd out of `priorTurns`, so record it here too.
            if let lastTurn = incoming.turns.last, lastTurn.role == .user {
                session.transcript.turns.append(lastTurn)
                session.transcript.turns.append(ChatMessage(role: .assistant, content: text))
                held = session
            }
        }
        return LLMResponse(text: text, finishReason: .stop)
    }

    /// Builds the held session for one conversation. The system prompt seeds `history[0]`
    /// rather than the `instructions:` parameter — `ChatSession` re-templates instructions
    /// into the KV stream on EVERY `respond()`, which would desync a held cache; a history
    /// `.system` turn is templated exactly once (and yields the same templated bytes).
    ///
    /// `ChatSession` is not `Sendable` (and not thread-safe), so it is created inside this
    /// `nonisolated` helper from `Sendable` inputs and handed back only inside the
    /// `@unchecked Sendable` `HeldChatSession` box — it never crosses the `InferenceActor`
    /// isolation boundary unboxed (which would trip the "sending risks data races" check).
    /// `ModelContainer` provides its own internal isolation.
    private nonisolated static func startSession(
        container: ModelContainer,
        systemPrompt: String,
        priorTurns: [ChatMessage],
        parameters: GenerateParameters,
        additionalContext: [String: any Sendable]?
    ) -> HeldChatSession {
        var history: [Chat.Message] = []
        if !systemPrompt.isEmpty {
            history.append(.system(systemPrompt))
        }
        history.append(contentsOf: priorTurns.map { message in
            switch message.role {
            case .assistant: return .assistant(message.content)
            case .user, .system: return .user(message.content)
            }
        })
        let session = ChatSession(
            container,
            instructions: nil,
            history: history,
            generateParameters: parameters,
            additionalContext: additionalContext
        )
        return HeldChatSession(
            session: session,
            transcript: SessionTranscript(systemPrompt: systemPrompt, turns: priorTurns))
    }

    /// Runs one generation on the boxed session. `generateParameters` and `additionalContext`
    /// are mutable vars on `ChatSession`, captured per `respond()` call — refreshing them here
    /// lets sampling and mode (`enable_thinking`) change across turns of a HELD session without
    /// forcing a rebuild. (Neither affects KV-cache structure: the cache-shaping knobs —
    /// kvBits/maxKVSize — are never set by this package.)
    private nonisolated static func respond(
        in held: HeldChatSession,
        to prompt: String,
        parameters: GenerateParameters,
        additionalContext: [String: any Sendable]?
    ) async throws -> String {
        held.session.generateParameters = parameters
        held.session.additionalContext = additionalContext
        let text = try await held.session.respond(to: prompt)
        // Cancellation seam (CAN-2): on cancel, mlx-swift-lm's generation loop stops per token
        // (Evaluate.swift checks Task.isCancelled every iteration) but `respond` RETURNS the
        // partial text instead of throwing. Convert that into the canonical CancellationError
        // HERE — before the caller records the turn and re-holds the session — so a cancelled
        // run never re-holds a KV cache with a truncated assistant turn (the caller's
        // take-the-box-out-first pattern leaves `held` nil when this throws).
        try Task.checkCancellation()
        return text
    }

    // MARK: - Structured output (contract 1.16.0, ENGINE-NEEDS N6)

    /// Grammar-constrained generation for `responseFormat` requests.
    ///
    /// Templates the full message list via the tokenizer chat template (`UserInput` →
    /// `UserInputProcessor.prepare`), then drives `TokenIterator` directly with
    /// `JSONConstrainedLogitProcessor` masking every token that can't extend a valid JSON
    /// prefix. Generation stops by construction: once the top-level value completes, the
    /// mask leaves only EOS.
    private func runStructured(container: ModelContainer,
                               request llm: LLMRequest,
                               parameters: GenerateParameters,
                               templateContext: [String: any Sendable]?) async throws
        -> LLMResponse {
        guard let format = llm.responseFormat else {
            throw PackageError.configurationMismatch(expected: "responseFormat", got: "nil")
        }

        // Contract format → grammar container. C12: defaults on both additive enums.
        let grammarContainer: JSONStateMachine.Container
        switch format {
        case .json(let container):
            switch container {
            case .object: grammarContainer = .object
            case .array:  grammarContainer = .array
            case .any:    grammarContainer = .any
            @unknown default: grammarContainer = .any
            }
        case .jsonSchema(let schema):
            // V1 best-effort lane (documented in the contract): valid-JSON syntax with the
            // container inferred from the schema root; field shape stays prompt-steered.
            grammarContainer = JSONSchemaHint.container(fromSchema: schema)
        @unknown default:
            throw PackageError.unsupportedRequestFeature(
                "responseFormat: unrecognized case (package built against an older contract)")
        }

        // Once-per-residency vocab classification (full-vocab pass over the tokenizer).
        if constrainedVocabulary == nil {
            constrainedVocabulary = await container.perform { context in
                var eosIDs = context.configuration.eosTokenIds
                if let id = context.tokenizer.eosTokenId { eosIDs.insert(id) }
                for token in context.configuration.extraEOSTokens {
                    if let id = context.tokenizer.convertTokenToId(token) { eosIDs.insert(id) }
                }
                return TokenVocabulary(
                    pieceForID: { context.tokenizer.convertIdToToken($0) },
                    eosTokenIDs: eosIDs)
            }
        }
        guard let vocabulary = constrainedVocabulary else { throw PackageError.notLoaded }

        // Same history mapping as the freeform path (system content hoisted to the front).
        // `Chat.Message` is not Sendable, so the transcript (which is) crosses into the
        // `perform` closure and the chat history is materialized inside it.
        let transcript = SessionTranscript(messages: llm.messages)

        // Thinking is disabled by default on structured calls: `<think>` tags can't be
        // emitted under the JSON mask anyway (special tokens are excluded), so leaving the
        // hybrid template in thinking mode would only fight the constraint. An explicit
        // `.thinking` mode still wins.
        let kwargs = templateContext ?? ["enable_thinking": false]

        // Unbounded-generation backstop: constrained decode force-stops at the complete
        // value (or on a grammar dead end), but a pathological in-string ramble could run
        // until the context window — cap it when the caller didn't.
        let maxTokens = parameters.maxTokens ?? 2048

        let (text, complete, stats): (String, Bool, JSONConstraintEngine.Stats) =
            try await container.perform { context in
                var history: [Chat.Message] = []
                if !transcript.systemPrompt.isEmpty {
                    history.append(.system(transcript.systemPrompt))
                }
                for turn in transcript.turns {
                    switch turn.role {
                    case .assistant: history.append(.assistant(turn.content))
                    case .user, .system: history.append(.user(turn.content))
                    }
                }
                let input = UserInput(chat: history, additionalContext: kwargs)
                let lmInput = try await context.processor.prepare(input: input)
                let engine = JSONConstraintEngine(vocabulary: vocabulary,
                                                  container: grammarContainer)
                let processor = JSONConstrainedLogitProcessor(engine: engine)
                var iterator = try TokenIterator(
                    input: lmInput, model: context.model, cache: nil,
                    processor: processor, sampler: parameters.sampler(),
                    maxTokens: maxTokens)
                var tokens: [Int] = []
                while let token = iterator.next() {
                    if vocabulary.eosTokenIDs.contains(token) { break }
                    tokens.append(token)
                    try Task.checkCancellation()   // C13: cooperatively evictable
                }
                return (context.tokenizer.decode(tokenIds: tokens),
                        engine.isComplete, engine.stats)
            }
        lastStructuredStats = stats

        // Honest finish reason: `.stop` only when the grammar really completed a top-level
        // value; a maxTokens truncation mid-value reports `.length` so the caller knows the
        // text may not parse.
        return LLMResponse(text: text, finishReason: complete ? .stop : .length)
    }

    /// Maps the canonical `Mode` tag onto Qwen3.5 chat-template kwargs. Qwen3.5 is a hybrid
    /// reasoner whose template emits `<think>…</think>` unless told otherwise:
    /// `.direct`/`.companion` disable thinking, `.thinking` enables it. A `nil` (or any other)
    /// mode returns `nil` so nothing is injected — keeping the change additive for callers that
    /// don't opt in. `enable_thinking` is the Qwen-family kwarg; other families would map their own.
    private nonisolated static func templateContext(for mode: Mode?) -> [String: any Sendable]? {
        guard let mode else { return nil }
        switch mode {
        case .direct, .companion: return ["enable_thinking": false]
        case .thinking:           return ["enable_thinking": true]
        default:                  return nil
        }
    }
}

/// The held cross-`run` `ChatSession` plus the exact transcript its KV cache encodes.
///
/// `@unchecked Sendable`: `ChatSession` is not `Sendable` (not thread-safe), but every touch is
/// serialized — the box lives on the `@InferenceActor` package, whose lifecycle the engine
/// drives in one serialization domain (C13), and the session inside is only used by the
/// package's `nonisolated` helpers, one call at a time. The box exists so the session reference
/// can be stored on the actor across runs and still be passed to those helpers without tripping
/// Swift 6's region-isolation "sending 'session' risks causing data races" check.
final class HeldChatSession: @unchecked Sendable {
    let session: ChatSession
    /// What the session's KV cache encodes; compared (exactly) against each incoming request
    /// by `SessionReuse.decision`. Mutated only on the `InferenceActor`.
    var transcript: SessionTranscript

    init(session: ChatSession, transcript: SessionTranscript) {
        self.session = session
        self.transcript = transcript
    }
}

extension QwenLLMPackage {
    /// The author one-liner the engine registers: manifest + license-gated factory. Read at
    /// registration time (outside the serialization domain), so `nonisolated`.
    public nonisolated static var registration: PackageRegistration {
        .of(QwenLLMPackage.self)
    }
}
