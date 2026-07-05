import Testing
import MLXToolKit
@testable import MLXQwenLLM

// Pure fingerprint tests for KV-cache reuse (`SessionReuse.decision`). The invariant under
// test: a cache hit is returned ONLY when the incoming request is exactly the held transcript
// plus one new user turn — false negatives merely cost a re-prefill, false positives would
// generate against wrong KV state and are forbidden.

private func msg(_ role: ChatMessage.Role, _ content: String) -> ChatMessage {
    ChatMessage(role: role, content: content)
}

private let system = "You are a concise assistant."

/// A held transcript after one completed exchange under `system`.
private let heldAfterOneExchange = SessionTranscript(
    systemPrompt: system,
    turns: [msg(.user, "Hi, I'm Ada."), msg(.assistant, "Hello Ada!")])

@Test func incomingTranscriptSplitsSystemFromTurns() {
    let t = SessionTranscript(messages: [
        msg(.system, "A"), msg(.user, "u1"), msg(.system, "B"), msg(.assistant, "a1"),
    ])
    #expect(t.systemPrompt == "A\n\nB")
    #expect(t.turns == [msg(.user, "u1"), msg(.assistant, "a1")])
}

@Test func hitOnExactAppendOfOneUserTurn() {
    let incoming = SessionTranscript(messages: [
        msg(.system, system),
        msg(.user, "Hi, I'm Ada."), msg(.assistant, "Hello Ada!"),
        msg(.user, "What's my name?"),
    ])
    #expect(SessionReuse.decision(held: heldAfterOneExchange, incoming: incoming)
            == .reuse(newUserTurn: "What's my name?"))
}

@Test func missWhenNothingHeld() {
    let incoming = SessionTranscript(messages: [msg(.user, "Hi")])
    #expect(SessionReuse.decision(held: nil, incoming: incoming) == .rebuild)
}

@Test func missOnEditedHistory() {
    let incoming = SessionTranscript(messages: [
        msg(.system, system),
        msg(.user, "Hi, I'm Eve."), msg(.assistant, "Hello Ada!"),   // user turn edited
        msg(.user, "What's my name?"),
    ])
    #expect(SessionReuse.decision(held: heldAfterOneExchange, incoming: incoming) == .rebuild)
}

@Test func missOnEditedAssistantTurn() {
    let incoming = SessionTranscript(messages: [
        msg(.system, system),
        msg(.user, "Hi, I'm Ada."), msg(.assistant, "Hello Ada"),    // reply not byte-identical
        msg(.user, "What's my name?"),
    ])
    #expect(SessionReuse.decision(held: heldAfterOneExchange, incoming: incoming) == .rebuild)
}

@Test func missOnChangedSystemPrompt() {
    let incoming = SessionTranscript(messages: [
        msg(.system, "You are a pirate."),
        msg(.user, "Hi, I'm Ada."), msg(.assistant, "Hello Ada!"),
        msg(.user, "What's my name?"),
    ])
    #expect(SessionReuse.decision(held: heldAfterOneExchange, incoming: incoming) == .rebuild)
}

@Test func missWhenLastTurnIsNotUser() {
    let incoming = SessionTranscript(messages: [
        msg(.system, system),
        msg(.user, "Hi, I'm Ada."), msg(.assistant, "Hello Ada!"),
        msg(.assistant, "And another thing —"),
    ])
    #expect(SessionReuse.decision(held: heldAfterOneExchange, incoming: incoming) == .rebuild)
}

@Test func missWhenMoreThanOneNewTurn() {
    let incoming = SessionTranscript(messages: [
        msg(.system, system),
        msg(.user, "Hi, I'm Ada."), msg(.assistant, "Hello Ada!"),
        msg(.user, "First."), msg(.user, "Second."),
    ])
    #expect(SessionReuse.decision(held: heldAfterOneExchange, incoming: incoming) == .rebuild)
}

@Test func missWhenHistoryShrinks() {
    let incoming = SessionTranscript(messages: [
        msg(.system, system),
        msg(.user, "Hi, I'm Ada."),
    ])
    #expect(SessionReuse.decision(held: heldAfterOneExchange, incoming: incoming) == .rebuild)
}

@Test func missOnIdenticalTranscriptWithNoNewTurn() {
    let incoming = SessionTranscript(
        systemPrompt: heldAfterOneExchange.systemPrompt,
        turns: heldAfterOneExchange.turns)
    #expect(SessionReuse.decision(held: heldAfterOneExchange, incoming: incoming) == .rebuild)
}

@Test func hitsChainAcrossAppendedExchanges() {
    // Simulate what run() does on a hit: append (user, assistant) and match again next turn.
    var held = heldAfterOneExchange
    held.turns.append(msg(.user, "What's my name?"))
    held.turns.append(msg(.assistant, "Ada."))
    let incoming = SessionTranscript(messages: [
        msg(.system, system),
        msg(.user, "Hi, I'm Ada."), msg(.assistant, "Hello Ada!"),
        msg(.user, "What's my name?"), msg(.assistant, "Ada."),
        msg(.user, "Spell it backwards."),
    ])
    #expect(SessionReuse.decision(held: held, incoming: incoming)
            == .reuse(newUserTurn: "Spell it backwards."))
}

@Test func hitWithEmptySystemPrompt() {
    let held = SessionTranscript(systemPrompt: "", turns: [msg(.user, "a"), msg(.assistant, "b")])
    let incoming = SessionTranscript(messages: [
        msg(.user, "a"), msg(.assistant, "b"), msg(.user, "c"),
    ])
    #expect(SessionReuse.decision(held: held, incoming: incoming) == .reuse(newUserTurn: "c"))
}
