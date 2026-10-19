import Foundation
import Testing
import MLXToolKit
import MLXServeCore
import MLXServeConformance
@testable import MLXQwenLLM

// The configured checkpoint as a `WeightSource` (AB-A-0195): the engine's offline MAT gate, the
// declaration per checkpoint, the MS-1 store-layout probe and resolution, and `needsDownload` on
// the repo that actually loads. No network, no weights: tiny probe files in a temp store.

private func tempStore() -> URL {
    FileManager.default.temporaryDirectory.appending(path: "qwen-store-\(UUID().uuidString)")
}

/// The hub-client layout this package's own download writes: `snapshots/<commit>/` behind `refs/main`.
private func writeSnapshot(_ repo: String, in root: URL, commit: String = "abc123") throws -> URL {
    let repoDir = ModelStore(root: root).directory(for: repo)!
    let snapshot = repoDir.appending(path: "snapshots/\(commit)")
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: repoDir.appending(path: "refs"), withIntermediateDirectories: true)
    try Data(commit.utf8).write(to: repoDir.appending(path: "refs/main"))
    FileManager.default.createFile(atPath: snapshot.appending(path: "config.json").path, contents: Data([0]))
    return snapshot
}

private let fourBInt4 = QwenModel(size: .b4, quant: .int4)

@Suite struct QwenMaterializationTests {

    // MARK: - Engine MAT gate, per published checkpoint

    @Test(arguments: QwenModel.allPublished)
    func matGatePassesEveryPublishedCheckpoint(model: QwenModel) {
        let report = MaterializationConformance.check(freshConfiguration: QwenLLMConfiguration(model: model))
        #expect(report.passed, "\(model.displayName):\n\(report.summary)")
    }

    // MARK: - Declaration

    @Test func declaresTheConfiguredCheckpointNotTheManifestDefault() {
        // The bug: the 4B int4 loaded, the marker landed on the manifest's 0.8B repo.
        let sources = QwenLLMConfiguration(model: fourBInt4).weightSources
        #expect(sources.map(\.role) == ["checkpoint"])
        #expect(sources[0].repo == "mlx-community/Qwen3.5-4B-MLX-4bit")
        #expect(sources[0].repo != QwenLLMPackage.manifest.provenance.sourceRepo)
        #expect(sources[0].matching == nil, "quant is tiered by repo: the whole snapshot")
        for model in QwenModel.allPublished {
            #expect(QwenLLMConfiguration(model: model).weightSources.map(\.repo) == [model.weightsRepo!])
        }
    }

    @Test func pinnedRevisionRidesTheSource() {
        #expect(QwenLLMConfiguration(revision: "abc123").weightSources[0].revision == "abc123")
        #expect(QwenLLMConfiguration().weightSources[0].revision == nil)
    }

    @Test func unpublishedQuantDeclaresNothing() {
        let cfg = QwenLLMConfiguration(model: QwenModel(size: .b4, quant: .int5))
        #expect(cfg.weightSources.isEmpty)
        #expect(cfg.resolvedModelDirectory(storeRoot: tempStore()) == nil)
    }

    // MARK: - Store layout (MS-1): probe and resolution

    @Test func hubSnapshotSatisfiesAndResolvesSnapshotFirst() throws {
        let root = tempStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = QwenLLMConfiguration(model: fourBInt4)
        #expect(cfg.missingWeightSources(storeRoot: root).count == 1)
        #expect(cfg.resolvedModelDirectory(storeRoot: root) == nil, "nothing there: load() downloads")
        let snapshot = try writeSnapshot("mlx-community/Qwen3.5-4B-MLX-4bit", in: root)
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty)
        #expect(cfg.resolvedModelDirectory(storeRoot: root)?.path == snapshot.path)
    }

    @Test func engineFlatLayoutSatisfiesAndResolves() throws {
        // Contract 1.24: the engine's materializer writes files directly under the repo directory.
        let root = tempStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = QwenLLMConfiguration(model: fourBInt4)
        let flat = ModelStore(root: root).directory(for: "mlx-community/Qwen3.5-4B-MLX-4bit")!
        try FileManager.default.createDirectory(at: flat, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: flat.appending(path: "config.json").path, contents: Data([0]))
        #expect(cfg.missingWeightSources(storeRoot: root).isEmpty)
        #expect(cfg.resolvedModelDirectory(storeRoot: root)?.path == flat.path)
    }

    @Test func aMarkerAloneIsNotACheckpoint() throws {
        // What the shared models folder held: `models--mlx-community--Qwen3.5-0.8B-MLX-8bit/` with
        // only `mlx-package.json`, stamped when the 4B loaded. It doesn't make the 0.8B present.
        let root = tempStore()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try writeSnapshot("mlx-community/Qwen3.5-4B-MLX-4bit", in: root)
        ModelStore(root: root).writeMarker(repo: "mlx-community/Qwen3.5-0.8B-MLX-8bit", revision: "main",
                                           capabilities: [.llm])
        let small = QwenLLMConfiguration()
        #expect(small.missingWeightSources(storeRoot: root).count == 1)
        #expect(small.resolvedModelDirectory(storeRoot: root) == nil)
        #expect(QwenLLMConfiguration(model: fourBInt4).missingWeightSources(storeRoot: root).isEmpty)
    }

    @Test func legacyNestedLayoutReadsAsMissing() throws {
        let root = tempStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = root.appending(path: "mlx-community/Qwen3.5-4B-MLX-4bit")
        try FileManager.default.createDirectory(at: legacy, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: legacy.appending(path: "config.json").path, contents: Data([0]))
        #expect(QwenLLMConfiguration(model: fourBInt4).missingWeightSources(storeRoot: root).count == 1)
    }

    // MARK: - The engine reads the checkpoint that loads

    @Test func needsDownloadProbesTheConfiguredCheckpoint() async throws {
        let root = tempStore()
        defer { try? FileManager.default.removeItem(at: root) }
        // Only the 0.8B's marker is in the store, as the shared folder had it. Before, that marker
        // answered for every checkpoint; now the 4B's own repo decides.
        ModelStore(root: root).writeMarker(repo: "mlx-community/Qwen3.5-0.8B-MLX-8bit", revision: "main",
                                           capabilities: [.llm])
        let engine = MLXServeEngine()
        await engine.useModelStore(ModelStore(root: root))
        try await engine.register(QwenLLMPackage.registration,
                                  configuration: QwenLLMConfiguration(model: fourBInt4, modelsRootDirectory: root))
        #expect(await engine.needsDownload(.llm), "the 0.8B's marker doesn't make the 4B present")
        _ = try writeSnapshot("mlx-community/Qwen3.5-4B-MLX-4bit", in: root)
        #expect(await engine.needsDownload(.llm) == false)
    }
}
