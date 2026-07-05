import MLXToolKit

/// The exact transcript a held `ChatSession`'s KV cache encodes: the (joined) system prompt
/// plus every non-system turn, in order, byte-for-byte as received — including the assistant
/// replies this package appended after generating them.
///
/// This is the fingerprint side of KV-cache reuse. It is deliberately **exact-string**: a cache
/// hit that isn't a true prefix of the incoming conversation would generate against wrong KV
/// state, so false positives are forbidden; a false negative merely costs a re-prefill.
struct SessionTranscript: Sendable, Equatable {
    /// All system turns joined with a blank line, in message order. Empty when the request
    /// carries no system message. Position within the message list is deliberately erased —
    /// the session hoists system content to the front of the KV stream either way (exactly as
    /// the pre-reuse code hoisted it into `instructions`).
    var systemPrompt: String
    /// Every non-system turn, in order, exactly as received.
    var turns: [ChatMessage]

    init(systemPrompt: String, turns: [ChatMessage]) {
        self.systemPrompt = systemPrompt
        self.turns = turns
    }

    /// Decompose an incoming request's message list into the fingerprint shape.
    init(messages: [ChatMessage]) {
        self.systemPrompt = messages
            .filter { $0.role == .system }
            .map(\.content)
            .joined(separator: "\n\n")
        self.turns = messages.filter { $0.role != .system }
    }
}

/// Pure decision logic for reusing a held `ChatSession` across `run(_:)` calls.
enum SessionReuse {
    enum Decision: Equatable {
        /// The held KV cache already encodes everything but this one new user turn — send
        /// only it, at the cache offset (`TokenIterator` does no prefix dedupe, so sending
        /// the full history on a hit would double-encode it).
        case reuse(newUserTurn: String)
        /// Anything else: rebuild a fresh session and re-prefill (the pre-reuse behavior,
        /// kept as the correctness fallback).
        case rebuild
    }

    /// Cache hit ⇔ same system prompt AND the incoming turns are the held transcript plus
    /// exactly one new user turn. Exact string equality only.
    static func decision(held: SessionTranscript?, incoming: SessionTranscript) -> Decision {
        guard let held,
              held.systemPrompt == incoming.systemPrompt,
              incoming.turns.count == held.turns.count + 1,
              let last = incoming.turns.last,
              last.role == .user,
              Array(incoming.turns.dropLast()) == held.turns
        else { return .rebuild }
        return .reuse(newUserTurn: last.content)
    }
}
