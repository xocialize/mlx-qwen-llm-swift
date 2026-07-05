// RunQwenLLM — live gates for the Qwen3.5 `llm` package (GPU inference runs here, not in the
// SPM test product, whose metallib is unreliable — the fleet CLI-gate convention).
//
//   swift run -c release RunQwenLLM --smoke                     one governed-shape generate
//   swift run -c release RunQwenLLM --kv-reuse                  KV-cache reuse gate:
//       (a) correctness A/B — the same temp-0 multi-turn conversation run with a fresh session
//           per turn (reuse off) vs the held session (reuse on). Asserted: hit/miss counters,
//           and that the HELD leg recalls facts from earlier turns (the reused cache really
//           encodes the prefix). Per-turn byte-match vs the fresh leg is reported but NOT
//           asserted: the held KV stream and a fresh re-template are not byte-identical BY
//           UPSTREAM DESIGN (e.g. `enable_thinking:false` puts an empty <think> block in the
//           generation prompt — cached; a re-template of the finished turn omits it), so a
//           temp-0 argmax near-tie can legitimately flip.
//       (b) per-turn latency table — rebuild re-prefills the whole transcript every turn, the
//           held session prefills only the new turn, so on-times must stay ~flat (asserted
//           against a long ~5k-token persona where re-prefill dominates);
//       (c) mismatch fallback — edit an earlier turn mid-conversation and verify the package
//           rebuilds (miss counter) and byte-matches a guaranteed-fresh run (both legs rebuild,
//           so THIS comparison is exact).
//   swift run -c release RunQwenLLM --mem-bench                 split-footprint measurement:
//       resident floor + activation peak at the documented ~2k-token envelope (phys_footprint,
//       the governor's basis), plus the held-KV retention drift across a reuse-on conversation.
//   swift run -c release RunQwenLLM --structured                structured-output gate (N6):
//       the three companion-shaped prompts (parseFacts → JSON array, parseAffect → JSON
//       object, decideSearchQuery → JSON object) × N=20 runs at temp 0.7. Asserted: 100%
//       strict-parse rate + correct top-level container + finishReason==.stop with
//       responseFormat. Reported: the measured baseline failure rate WITHOUT responseFormat
//       (companion-style bracket-scrape parsing) and the per-step masking latency overhead.
//
//   Optional: --models-root <dir> to point the Hub cache somewhere other than the default
//   (~/.cache/huggingface/hub, where a prior download already lives on the dev box).

import Foundation
import MLX
import MLXToolKit
import MLXQwenLLM

func gbOf(_ b: UInt64) -> Double { Double(b) / 1_000_000_000.0 }

/// OS `phys_footprint` via `task_info(TASK_VM_INFO)` — the MemoryGovernor's basis.
func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

/// Background phys high-water sampler (the peak is a transient inside prefill/decode).
final class PhysSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var _max: UInt64 = 0
    private var _running = false
    func start() {
        lock.lock(); _running = true; lock.unlock()
        let t = Thread { [weak self] in
            while self?.running == true {
                self?.observe(physFootprintBytes())
                Thread.sleep(forTimeInterval: 0.025)
            }
        }
        t.stackSize = 1 << 20
        t.start()
    }
    var running: Bool { lock.lock(); defer { lock.unlock() }; return _running }
    func observe(_ p: UInt64) { lock.lock(); if p > _max { _max = p }; lock.unlock() }
    func resetMax() { lock.lock(); _max = physFootprintBytes(); lock.unlock() }
    func maxBytes() -> UInt64 { lock.lock(); defer { lock.unlock() }; return _max }
    func stop() { lock.lock(); _running = false; lock.unlock() }
}

/// A companion-shaped persona at conversational scale (correctness leg — the 0.8B model must
/// stay smart enough to recall facts through it).
let personaSystem = "You are Roxy, a warm, concise on-device companion. You answer in one or "
    + "two short sentences and remember what the user tells you."

/// The same persona padded to ~5k tokens (latency leg — long enough that re-prefilling it
/// every turn dominates the rebuild path, the signal the latency table is built to show).
let longPersonaSystem = String(
    repeating: "You are Roxy, a warm, concise on-device companion. You answer in one or two "
        + "short sentences, remember what the user tells you, and never invent facts about them. ",
    count: 96)

let gateUserTurns = [
    "Hi! My name is Marisol and I live in Reykjavík.",
    "What city do I live in? Answer in one short sentence.",
    "What is my first name? Answer in one short sentence.",
    "Summarize everything you know about me in one sentence.",
]

/// Runs the gate conversation start-to-finish on `pkg`, threading the transcript exactly the
/// way a consuming app does (history so far + the package's own prior replies + one new turn).
@InferenceActor
func runConversation(
    _ pkg: QwenLLMPackage, reuse: Bool, system: String, userTurns: [String], maxTokens: Int
) async throws -> (texts: [String], secs: [Double]) {
    pkg.kvCacheReuseEnabled = reuse
    let params = LLMParameters(temperature: 0, maxTokens: maxTokens)
    var messages: [ChatMessage] = [ChatMessage(role: .system, content: system)]
    var texts: [String] = []
    var secs: [Double] = []
    for turn in userTurns {
        messages.append(ChatMessage(role: .user, content: turn))
        let t0 = Date()
        let resp = try await pkg.run(
            LLMRequest(messages: messages, parameters: params, mode: .direct)) as! LLMResponse
        secs.append(Date().timeIntervalSince(t0))
        texts.append(resp.text)
        messages.append(ChatMessage(role: .assistant, content: resp.text))
    }
    return (texts, secs)
}

@InferenceActor
func smoke(cfg: QwenLLMConfiguration) async throws {
    let pkg = QwenLLMPackage(configuration: cfg)
    let t0 = Date()
    try await pkg.load()
    print(String(format: "[smoke] load %.1fs", Date().timeIntervalSince(t0)))
    let r0 = Date()
    let resp = try await pkg.run(
        LLMRequest(
            messages: [
                ChatMessage(role: .system, content: personaSystem),
                ChatMessage(role: .user, content: "Say hello in one short sentence."),
            ],
            parameters: LLMParameters(temperature: 0.7, topP: 0.95, maxTokens: 128),
            mode: .direct)) as! LLMResponse
    print(String(format: "[smoke] run %.1fs · %d chars", Date().timeIntervalSince(r0), resp.text.count))
    print("[smoke] ---\n\(resp.text.prefix(400))\n[smoke] ---")
    await pkg.unload()
    print(resp.text.isEmpty ? "[smoke] FAIL ❌ (empty)" : "[smoke] PASS ✅")
}

@InferenceActor
func kvReuseGate(cfg: QwenLLMConfiguration) async throws {
    var failures: [String] = []
    let pkg = QwenLLMPackage(configuration: cfg)
    try await pkg.load()

    // Warmup: kernel compile, excluded from every timing below.
    _ = try await pkg.run(
        LLMRequest(prompt: "Hi", parameters: LLMParameters(temperature: 0, maxTokens: 8),
                   mode: .direct))

    // (a) Correctness A/B: identical temp-0 conversation, fresh-per-turn vs held. Byte-match
    // is informational (see header); the assertions are the counters and that the held leg
    // RECALLS earlier-turn facts through the reused cache.
    print("[kv-reuse] correctness leg — baseline pass (reuse OFF — fresh session per turn)…")
    let off = try await runConversation(
        pkg, reuse: false, system: personaSystem, userTurns: gateUserTurns, maxTokens: 120)
    var (hits0, misses0) = (pkg.kvReuseHits, pkg.kvReuseMisses)
    print("[kv-reuse] correctness leg — held pass (reuse ON)…")
    let on = try await runConversation(
        pkg, reuse: true, system: personaSystem, userTurns: gateUserTurns, maxTokens: 120)
    let hits = pkg.kvReuseHits - hits0
    let misses = pkg.kvReuseMisses - misses0

    print("[kv-reuse] turn |   off(s) |    on(s) | byte-match (informational)")
    for i in 0..<gateUserTurns.count {
        let match = off.texts[i] == on.texts[i]
        print(String(format: "[kv-reuse]  %2d  | %8.2f | %8.2f | %@",
                     i + 1, off.secs[i], on.secs[i], match ? "same" : "DIVERGED"))
        if !match {
            print("[kv-reuse]    off: \(off.texts[i].prefix(200))")
            print("[kv-reuse]     on: \(on.texts[i].prefix(200))")
        }
    }
    print("[kv-reuse] held-pass counters: hits=\(hits) misses=\(misses) "
          + "(expected hits=\(gateUserTurns.count - 1) misses=1)")
    if hits != gateUserTurns.count - 1 || misses != 1 {
        failures.append("unexpected hit/miss counts (hits=\(hits) misses=\(misses))")
    }
    // Recall through the held cache: turn 2 asked for the city, turn 3 for the name — both
    // stated only in turn 1, which on the held leg exists ONLY inside the reused KV cache.
    let recalls: [(turn: Int, expect: String)] = [(2, "Reykjavík"), (3, "Marisol")]
    for (turn, expect) in recalls {
        let text = on.texts[turn - 1]
        let ok = text.localizedCaseInsensitiveContains(expect)
        print("[kv-reuse] held-leg recall (turn \(turn) ⊇ \"\(expect)\"): \(ok ? "✅" : "❌") — \(text.prefix(120))")
        if !ok { failures.append("held leg failed to recall \"\(expect)\" at turn \(turn)") }
    }

    // (b) Latency leg: a ~5k-token persona so the rebuild path re-prefills ~5k+ tokens every
    // turn while the held path prefills only the new turn. Hit turns must be well under the
    // rebuild turns (conservative 0.6× on the post-first-turn average).
    print("[kv-reuse] latency leg (~5k-token persona) — reuse OFF…")
    let lOff = try await runConversation(
        pkg, reuse: false, system: longPersonaSystem, userTurns: gateUserTurns, maxTokens: 24)
    (hits0, misses0) = (pkg.kvReuseHits, pkg.kvReuseMisses)
    print("[kv-reuse] latency leg — reuse ON…")
    let lOn = try await runConversation(
        pkg, reuse: true, system: longPersonaSystem, userTurns: gateUserTurns, maxTokens: 24)
    let lHits = pkg.kvReuseHits - hits0
    print("[kv-reuse] turn |   off(s) |    on(s)")
    for i in 0..<gateUserTurns.count {
        print(String(format: "[kv-reuse]  %2d  | %8.2f | %8.2f%@",
                     i + 1, lOff.secs[i], lOn.secs[i], i == 0 ? "  (miss — first turn prefills)" : ""))
    }
    let offAvg = lOff.secs.dropFirst().reduce(0, +) / Double(lOff.secs.count - 1)
    let onAvg = lOn.secs.dropFirst().reduce(0, +) / Double(lOn.secs.count - 1)
    print(String(format: "[kv-reuse] post-first-turn avg: off %.2fs · on %.2fs (%.1f× speedup)",
                 offAvg, onAvg, offAvg / max(onAvg, 0.001)))
    if lHits != gateUserTurns.count - 1 {
        failures.append("latency leg expected \(gateUserTurns.count - 1) hits, got \(lHits)")
    }
    if onAvg >= offAvg * 0.6 {
        failures.append(String(format: "hit turns not meaningfully faster (on %.2fs vs off %.2fs)", onAvg, offAvg))
    }

    // (c) Mismatch fallback: edit an earlier user turn mid-conversation → the fingerprint must
    // reject the held cache (rebuild) and the answer must equal a guaranteed-fresh run.
    var edited: [ChatMessage] = [ChatMessage(role: .system, content: personaSystem)]
    for (i, turn) in gateUserTurns.enumerated() {
        let content = i == 0 ? "Hi! My name is Gertrude and I live in Oslo." : turn
        edited.append(ChatMessage(role: .user, content: content))
        edited.append(ChatMessage(role: .assistant, content: on.texts[i]))
    }
    edited.append(ChatMessage(role: .user, content: "What is my name? One short sentence."))
    let editedRequest = LLMRequest(
        messages: edited, parameters: LLMParameters(temperature: 0, maxTokens: 120), mode: .direct)

    let missesBefore = pkg.kvReuseMisses
    pkg.kvCacheReuseEnabled = true
    let fallback = try await pkg.run(editedRequest) as! LLMResponse
    let fellBack = pkg.kvReuseMisses == missesBefore + 1
    pkg.kvCacheReuseEnabled = false
    let fresh = try await pkg.run(editedRequest) as! LLMResponse
    print("[kv-reuse] mismatch fallback: rebuild=\(fellBack ? "✅" : "❌") "
          + "output-match=\(fallback.text == fresh.text ? "✅" : "❌")")
    print("[kv-reuse]    edited-history answer: \(fallback.text.prefix(200))")
    if !fellBack { failures.append("edited history did not fall back to a rebuild") }
    if fallback.text != fresh.text {
        print("[kv-reuse]    fresh: \(fresh.text.prefix(200))")
        failures.append("edited-history output diverged from a fresh run")
    }

    await pkg.unload()
    if failures.isEmpty {
        print("[kv-reuse] PASS ✅")
    } else {
        print("[kv-reuse] FAIL ❌")
        for f in failures { print("[kv-reuse]   - \(f)") }
        exit(1)
    }
}

@InferenceActor
func memBench(cfg: QwenLLMConfiguration) async throws {
    print("[mem-bench] \(cfg.model.displayName) · envelope \(QwenModel.contextEnvelopeTokens) tokens")
    let sampler = PhysSampler(); sampler.start()
    let pkg = QwenLLMPackage(configuration: cfg)
    let t0 = Date()
    try await pkg.load()
    print(String(format: "[mem-bench] load %.1fs  phys-after-load=%.2f GB",
                 Date().timeIntervalSince(t0), gbOf(physFootprintBytes())))

    // Warmup (kernel compile; excluded from the peak), then the floor.
    _ = try await pkg.run(
        LLMRequest(prompt: "Hi", parameters: LLMParameters(temperature: 0, maxTokens: 16),
                   mode: .direct))
    await pkg.unload()          // also drops the held session so the floor is weights-only
    try await pkg.load()
    Memory.clearCache()
    let floor = physFootprintBytes()
    print(String(format: "[mem-bench] resident floor (post-warmup + clearCache): %.2f GB", gbOf(floor)))

    // Envelope run (single-shot, KV-reuse irrelevant on turn 1): ~1.2k-token prefill +
    // generation toward the documented ~2k total context.
    let envelopeSystem = String(
        repeating: "You are Roxy, a warm, concise on-device companion. You answer in one or "
            + "two short sentences, remember what the user tells you, and never invent facts. ",
        count: 24)
    sampler.resetMax()
    let r0 = Date()
    let resp = try await pkg.run(
        LLMRequest(
            messages: [
                ChatMessage(role: .system, content: envelopeSystem),
                ChatMessage(role: .user, content: "Tell me a long story about Reykjavík."),
            ],
            parameters: LLMParameters(temperature: 0.7, topP: 0.95, maxTokens: 800),
            mode: .direct)) as! LLMResponse
    let peak = sampler.maxBytes()
    let activation = peak > floor ? peak - floor : 0
    print(String(format: "[mem-bench] envelope run %.1fs · %d chars", Date().timeIntervalSince(r0), resp.text.count))
    print(String(format: "[mem-bench] SPLIT floor=%.2f GB  peak=%.2f GB  act=%.2f GB  (declared act=%.2f GB)",
                 gbOf(floor), gbOf(peak), gbOf(activation), gbOf(cfg.model.peakActivationBytes)))

    // Held-KV retention drift: a reuse-on conversation retains KV/recurrent state by design —
    // this shows how much rides along per turn (the "active climbing across turns" number).
    let base = physFootprintBytes()
    let drift = try await runConversation(
        pkg, reuse: true, system: personaSystem, userTurns: gateUserTurns, maxTokens: 120)
    print(String(format: "[mem-bench] reuse-on conversation: %d turns in %.1fs total",
                 gateUserTurns.count, drift.secs.reduce(0, +)))
    print(String(format: "[mem-bench] held-session retention after conversation: %+.0f MB phys",
                 (Double(physFootprintBytes()) - Double(base)) / 1_000_000.0))
    await pkg.unload()
    Memory.clearCache()
    print(String(format: "[mem-bench] post-unload phys: %.2f GB", gbOf(physFootprintBytes())))
    sampler.stop()
}

/// The three MLXCompanion call-site shapes that motivated N6, expressed as responseFormat
/// requests (the exact consumer recipes documented in the ENGINE-NEEDS closure note).
struct StructuredCase {
    let name: String
    let system: String
    let user: String
    let format: ResponseFormat
    /// "object" / "array" — the required top-level container of the parsed value.
    let expects: String
}

let structuredCases: [StructuredCase] = [
    StructuredCase(
        name: "parseFacts",
        system: "You extract stable facts about the user from conversation. Respond with ONLY "
            + "a JSON array of short fact strings. No prose, no code fences.",
        user: "User said: \"I'm Marisol, I live in Reykjavík with my two cats, and I teach "
            + "piano on weekends. Lately I've been learning Icelandic.\" Extract the facts.",
        format: .json(container: .array),
        expects: "array"),
    StructuredCase(
        name: "parseAffect",
        system: "You read the user's emotional state. Respond with ONLY a JSON object of the "
            + "shape {\"mood\": string, \"energy\": number 0..1, \"valence\": number -1..1}. "
            + "No prose.",
        user: "User said: \"honestly today was a lot. the recital went fine I guess but I'm "
            + "completely wiped and kind of on edge.\" Read their affect.",
        format: .json(container: .object),
        expects: "object"),
    StructuredCase(
        name: "decideSearchQuery",
        system: "You decide whether answering needs a web search. Respond with ONLY a JSON "
            + "object: {\"action\": \"search\", \"query\": \"<terms>\"} if current information "
            + "is needed, or {\"action\": \"none\"} if not. No prose.",
        user: "User asked: \"what's the weather looking like in Reykjavík this weekend?\"",
        format: .json(container: .object),
        expects: "object"),
]

/// Companion-style salvage parse for the BASELINE leg (mirrors the regex-scrape the N6 call
/// sites do today): try the trimmed text, then the outermost bracket span.
func scrapeJSON(_ text: String, expects: String) -> Bool {
    let (open, close): (Character, Character) = expects == "array" ? ("[", "]") : ("{", "}")
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    var candidates = [trimmed]
    if let a = trimmed.firstIndex(of: open), let b = trimmed.lastIndex(of: close), a < b {
        candidates.append(String(trimmed[a...b]))
    }
    for candidate in candidates {
        guard let data = candidate.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else { continue }
        if expects == "array", parsed is [Any] { return true }
        if expects == "object", parsed is [String: Any] { return true }
    }
    return false
}

/// Strict parse for the CONSTRAINED leg: the entire response must be one valid JSON value of
/// the requested container. No salvage.
func strictParse(_ text: String, expects: String) -> Bool {
    guard let data = text.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
          let parsed = try? JSONSerialization.jsonObject(with: data) else { return false }
    if expects == "array" { return parsed is [Any] }
    return parsed is [String: Any]
}

@InferenceActor
func structuredGate(cfg: QwenLLMConfiguration, runs: Int) async throws {
    var failures: [String] = []
    let pkg = QwenLLMPackage(configuration: cfg)
    try await pkg.load()

    // Warmup both paths (kernel compile + the once-per-residency vocab classification),
    // excluded from every timing below.
    _ = try await pkg.run(
        LLMRequest(prompt: "Hi", parameters: LLMParameters(temperature: 0, maxTokens: 8),
                   mode: .direct))
    let v0 = Date()
    _ = try await pkg.run(
        LLMRequest(prompt: "Say hi as JSON.",
                   parameters: LLMParameters(temperature: 0, maxTokens: 32),
                   mode: .direct, responseFormat: .json))
    print(String(format: "[structured] warmup (incl. one-time vocab classification): %.2fs",
                 Date().timeIntervalSince(v0)))

    let params = LLMParameters(temperature: 0.7, topP: 0.95, maxTokens: 256)
    var totalMaskSeconds = 0.0
    var totalMaskSteps = 0

    for c in structuredCases {
        var baselineOK = 0
        var constrainedOK = 0
        var baselineSecs = 0.0
        var constrainedSecs = 0.0

        for _ in 0..<runs {
            let messages = [ChatMessage(role: .system, content: c.system),
                            ChatMessage(role: .user, content: c.user)]

            // Baseline leg: today's behavior — no responseFormat, companion-style scrape.
            var t0 = Date()
            let free = try await pkg.run(
                LLMRequest(messages: messages, parameters: params, mode: .direct)) as! LLMResponse
            baselineSecs += Date().timeIntervalSince(t0)
            if scrapeJSON(free.text, expects: c.expects) { baselineOK += 1 }

            // Constrained leg: strict parse, correct container, honest finishReason.
            t0 = Date()
            let structured = try await pkg.run(
                LLMRequest(messages: messages, parameters: params, mode: .direct,
                           responseFormat: c.format)) as! LLMResponse
            constrainedSecs += Date().timeIntervalSince(t0)
            if strictParse(structured.text, expects: c.expects),
               structured.finishReason == .stop {
                constrainedOK += 1
            } else {
                print("[structured]   ✗ \(c.name) constrained miss "
                      + "(finish=\(String(describing: structured.finishReason))): "
                      + "\(structured.text.prefix(160))")
            }
            if let stats = pkg.lastStructuredStats {
                totalMaskSeconds += stats.maskSeconds
                totalMaskSteps += stats.steps
            }
        }

        print(String(format: "[structured] %-18s baseline(scrape) %2d/%d · constrained(strict) %2d/%d · avg %.2fs vs %.2fs",
                     (c.name as NSString).utf8String!, baselineOK, runs, constrainedOK, runs,
                     baselineSecs / Double(runs), constrainedSecs / Double(runs)))
        if constrainedOK != runs {
            failures.append("\(c.name): constrained parse rate \(constrainedOK)/\(runs) (must be 100%)")
        }
    }

    if totalMaskSteps > 0 {
        let perStepMs = totalMaskSeconds / Double(totalMaskSteps) * 1000
        print(String(format: "[structured] masking overhead: %.3f ms/step over %d steps "
                     + "(decode step on 0.8B-8bit is ~5-15 ms — target: negligible)",
                     perStepMs, totalMaskSteps))
        if perStepMs > 5 {
            failures.append(String(format: "per-step masking overhead %.2f ms is not negligible", perStepMs))
        }
    }

    await pkg.unload()
    if failures.isEmpty {
        print("[structured] PASS ✅")
    } else {
        print("[structured] FAIL ❌")
        for f in failures { print("[structured]   - \(f)") }
        exit(1)
    }
}

let args = CommandLine.arguments
var modelsRoot: URL? = nil
if let i = args.firstIndex(of: "--models-root"), i + 1 < args.count {
    modelsRoot = URL(fileURLWithPath: args[i + 1])
}
let cfg = QwenLLMConfiguration(model: .default, modelsRootDirectory: modelsRoot)

if args.contains("--kv-reuse") {
    try await kvReuseGate(cfg: cfg)
} else if args.contains("--mem-bench") {
    try await memBench(cfg: cfg)
} else if args.contains("--structured") {
    var runs = 20
    if let i = args.firstIndex(of: "--runs"), i + 1 < args.count, let n = Int(args[i + 1]) {
        runs = n
    }
    try await structuredGate(cfg: cfg, runs: runs)
} else if args.contains("--smoke") {
    try await smoke(cfg: cfg)
} else {
    print("usage: RunQwenLLM --smoke | --kv-reuse | --mem-bench | --structured [--runs N]  [--models-root <dir>]")
    print("  model: \(QwenModel.default.displayName) via \(QwenModel.default.weightsRepo ?? "?")")
    print("  weights default to the standard Hub cache (~/.cache/huggingface/hub)")
}
