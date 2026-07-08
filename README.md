# Go Agent Core

Standalone Go prototype for an agent runtime inspired by `@earendil-works/pi-agent-core`.

The current implementation focuses on provider-neutral messages, streaming provider contracts, tool contracts, lifecycle events, and a low-level loop that can stream assistant output, execute tool calls, append tool results, and continue until the model stops requesting tools.

## Status

Implemented:

- Go module scaffold using `go 1.26`
- Provider-neutral message types in `llm`
- Provider request and stream event types in `llm`
- Tool execution contracts in `tool`
- Low-level agent loop and lifecycle events in `agent`
- OpenRouter streaming provider in `providers/openrouter`
- Parallel tool execution
- Fake-provider tests for streaming, tool calls, tool result ordering, and tool errors
- JSONL session persistence
- Minimal example CLI in `examples/simple`

Not implemented yet:

- OpenRouter integration smoke tests

See [PLAN.md](PLAN.md) for the full implementation plan.

## Packages

- `agent`: low-level loop and lifecycle events
- `llm`: provider-neutral messages, requests, and provider stream events
- `tool`: tool metadata, calls, progress updates, and execution results
- `providers/openrouter`: OpenRouter chat-completions streaming provider
- `session`: JSONL transcript persistence

## Quick Start

Run the example CLI:

```sh
go run ./examples/simple "Say hi"
```

Run tests:

```sh
go test ./...
```

Use OpenRouter by setting an API key:

```sh
export OPENROUTER_API_KEY="..."
```

The provider defaults to OpenRouter's `deepseek/deepseek-v4-flash` model. Import `go-agent-core/providers/openrouter` and pass `openrouter.Config{Model: "provider/model"}` to use another OpenRouter model slug.

## Current Loop Shape

The low-level loop is exposed as:

```go
messages, err := agent.RunLoop(ctx, transcript, "hello", agent.LoopOptions{
    Provider: provider,
    Tools:    tools,
    OnEvent:  handleEvent,
})
```

`RunLoop` appends the user prompt, streams assistant events from the provider, finalizes an assistant message, executes requested tools, appends tool result messages, and repeats until there are no tool calls.

For persistent conversations, use the stateful wrapper:

```go
runtime := agent.NewAgent(agent.AgentOptions{
    Provider: provider,
    Tools:    tools,
})

unsubscribe := runtime.Subscribe(handleEvent)
defer unsubscribe()

if err := runtime.Prompt(ctx, "hello"); err != nil {
    // Handle provider, cancellation, or loop errors.
}
```

`Agent` serializes prompts, owns the transcript, exposes copied state snapshots, and supports `Abort`, `Wait`, `Steer`, and `FollowUp` for active conversations.
