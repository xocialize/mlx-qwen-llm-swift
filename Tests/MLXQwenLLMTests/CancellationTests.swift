// CancellationTests.swift — Qwen LLM through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before notLoaded
// validation or weights). CAN-3 is the document of record for the checkpoint cadence: BOTH
// generation paths bail per generated token — the plain (held-ChatSession) path via
// mlx-swift-lm's own per-token `Task.isCancelled` check plus the wrapper's post-respond
// `try Task.checkCancellation()` (which also guarantees a cancelled run never re-holds a
// truncated session), and the structured path via the package-owned TokenIterator drive.

import Foundation
import MLXServeConformance
import MLXToolKit
import Testing
@testable import MLXQwenLLM

// MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

@Test func canGatePreCancelledRun() async {
    // Stub config; construction is cheap (C13) and the entry checkpoint throws before
    // validation or weights are touched, so this is offline-safe.
    let package = QwenLLMPackage(configuration: QwenLLMConfiguration())
    let report = await CancellationConformance.checkRun(
        package: package,
        request: LLMRequest(prompt: "probe"))
    #expect(report.passed, "\(report.summary)")
}

// MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

@Test func canCadenceDeclaration() {
    // llm is not a long-run capability, but the declared peak activation (~2.2 GB prefill
    // scratch at the 2k envelope on 0.8B; width-scaled higher on 4B/9B) crosses the 2 GB
    // threshold, so the sub-second exemption is not available.
    #expect(CancellationConformance.longRunImplied(by: QwenLLMPackage.manifest))

    let report = CancellationConformance.checkCadence(
        manifest: QwenLLMPackage.manifest,
        posture: .cadence([
            // Per generated token, on both paths:
            // — Plain path (held ChatSession): mlx-swift-lm 3.31.4's generation loop checks
            //   `Task.isCancelled` once per token (MLXLMCommon/Evaluate.swift, `tokenLoop`)
            //   and stops; the wrapper's post-respond `try Task.checkCancellation()`
            //   (QwenLLMPackage.respond(in:to:parameters:additionalContext:)) converts the
            //   silent partial return into the canonical CancellationError before the session
            //   could be re-held.
            // — Structured path: the package-owned TokenIterator drive checks
            //   `try Task.checkCancellation()` once per generated token
            //   (QwenLLMPackage.runStructured, the `while let token = iterator.next()` loop).
            .init(phase: .generate, unit: .token),
        ]))
    #expect(report.passed, "\(report.summary)")
}
