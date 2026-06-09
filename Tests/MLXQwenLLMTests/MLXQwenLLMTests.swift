import Testing
import MLXToolKit
@testable import MLXQwenLLM

@Test func manifestExposesLLMAndIsPermissive() {
    let manifest = QwenLLMPackage.manifest
    #expect(manifest.capabilities.contains(.llm))
    #expect(LicensePolicy.permissiveOnly.evaluate(manifest.license).isAdmitted)
    #expect(manifest.surfaces.count == 1)
}

@Test func repoIdsFollowMLXCommunityConvention() {
    #expect(QwenModel.default == QwenModel(size: .b0_8, quant: .int8))
    #expect(QwenModel.default.weightsRepo == "mlx-community/Qwen3.5-0.8B-MLX-8bit")
    #expect(QwenModel(size: .b9, quant: .int4).weightsRepo == "mlx-community/Qwen3.5-9B-MLX-4bit")
    #expect(QwenModel(size: .b4, quant: .bf16).weightsRepo == "mlx-community/Qwen3.5-4B-MLX-bf16")
    // fp16 isn't published under the mlx-community suffix scheme.
    #expect(QwenModel(size: .b4, quant: .fp16).weightsRepo == nil)
}

@Test func registrationConstructsPackage() throws {
    // Construct only — load()/run() now download ~1 GB and require Metal, so they belong in a
    // device/integration test, not the unit suite.
    let package = try QwenLLMPackage.registration.makePackage(QwenLLMConfiguration())
    #expect(package is QwenLLMPackage)
}

@Test func runRejectsUnloaded() async throws {
    let package = try QwenLLMPackage.registration.makePackage(QwenLLMConfiguration())
    await #expect(throws: PackageError.self) {
        _ = try await package.run(LLMRequest(prompt: "Hello"))
    }
}
