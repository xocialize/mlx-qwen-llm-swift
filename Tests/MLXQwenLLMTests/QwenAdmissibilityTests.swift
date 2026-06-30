import Testing
import MLXToolKit
import MLXServeCore
@testable import MLXQwenLLM

// Sanity markers: run the Qwen3.5 catalog through the engine's admissibility check to see what a
// given machine can load. Pure logic (no MLX at runtime) against DeviceProfile + MemoryGovernor.

private func engine(budgetBytes: UInt64) -> MLXServeEngine {
    let device = DeviceProfile(
        chipTier: .max,
        macOS: SemanticVersion(major: 26, minor: 0, patch: 0),
        backends: [.metalGPU],
        totalMemoryBytes: 64_000_000_000
    )
    return MLXServeEngine(device: device, governor: MemoryGovernor(budgetBytes: budgetBytes))
}

@Test func catalogIsTheKnownPublishedSet() {
    #expect(QwenModel.allPublished.count == 7) // 0.8B & 4B ×{4,8,bf16} + 9B/4bit
    #expect(QwenModel.allPublished.contains(.default)) // 0.8B 8-bit
    #expect(QwenModel.allPublished.allSatisfy { $0.weightsRepo != nil })
}

@Test func requirementsDeriveFromFootprint() {
    let model = QwenModel(size: .b9, quant: .int4)
    #expect(model.requirements.footprints.first?.residentBytes == model.residentBytes)
    #expect(model.requirements.requiredBackends == [.metalGPU])
}

@Test func allVariantsFitOnLargeBudget() async {
    let e = engine(budgetBytes: 16_000_000_000) // 16 GB
    for model in QwenModel.allPublished {
        let verdict = await e.admissibility(for: model.requirements)
        #expect(verdict.admissible, "\(model.displayName) should be admissible at 16 GB")
    }
}

@Test func tinyBudgetOnlyAdmitsSmallest() async {
    // 4 GB budget. With the split footprint (weights + measured prefill-scratch transient at the
    // 2048-token envelope), the admission charge is `residentBytes + peakActivationBytes`. The three
    // 0.8B variants total 2.6/3.0/3.8 GB (all < 4 GB); the smallest 4B (int4) is 7.6 GB (> 4 GB), so
    // only the 0.8B family fits. (Pre-split this test used 2 GB against a flat `onDisk + 600 MB`
    // resident with a zero transient — the split makes the transient honest, so the budget moves up.)
    let e = engine(budgetBytes: 4_000_000_000) // 4 GB
    var admissible: Set<QwenModel> = []
    for model in QwenModel.allPublished where await e.admissibility(for: model.requirements).admissible {
        admissible.insert(model)
    }
    #expect(admissible == [QwenModel(size: .b0_8, quant: .int4),
                           QwenModel(size: .b0_8, quant: .int8),
                           QwenModel(size: .b0_8, quant: .bf16)])

    // The 9B 4-bit is the clearest "won't load on a small machine" marker — eligible on the
    // device, just doesn't fit the memory budget.
    let nineB = await e.admissibility(for: QwenModel(size: .b9, quant: .int4).requirements)
    #expect(!nineB.admissible)
    #expect(nineB.eligibility.isEligible)
}
