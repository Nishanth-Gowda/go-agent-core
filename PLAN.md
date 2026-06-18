# Go Agent Core Plan

## Summary

Build a standalone Go implementation that mimics the core capabilities of `@earendil-works/pi-agent-core`, using OpenRouter as the only v1 provider.

The implementation target is capability parity, not TypeScript API parity: stateful agent runtime, streaming assistant messages, tool calls, event lifecycle, queues, cancellation, and tests.

## Key Changes

- Create a new standalone Go project at `/Users/nishanthgowda/Developer/go-agent-core`.
- Use `module go-agent-core` for the initial local prototype.
- Plan packages:
  - `agent`: stateful `Agent`, loop, lifecycle events, queues.
  - `llm`: provider-neutral messages, model requests, streaming interface.
  - `openrouter`: OpenRouter chat-completions streaming provider.
  - `tool`: tool definition, JSON schema metadata, execution results.
  - `session`: in-memory transcript first; JSONL persistence later.
- Use OpenRouter only for v1:
  - API key from `OPENROUTER_API_KEY`.
  - Streaming via OpenRouter chat completions API.
  - Tool calls normalized into internal `ToolCall` structs.
- Keep v1 minimal:
  - No multi-provider registry.
  - No compaction.
  - No skills/prompt templates.
  - No durable harness recovery.
  - No browser proxy.

## Public API Shape

The v1 API should expose these interfaces:

```go
type Agent struct {
    State AgentState
}

func NewAgent(opts AgentOptions) *Agent
func (a *Agent) Prompt(ctx context.Context, text string) error
func (a *Agent) Continue(ctx context.Context) error
func (a *Agent) Abort()
func (a *Agent) Wait() error
func (a *Agent) Subscribe(fn EventHandler) func()
func (a *Agent) Steer(msg Message)
func (a *Agent) FollowUp(msg Message)
```

```go
type Provider interface {
    Stream(ctx context.Context, req Request) (<-chan ProviderEvent, error)
}
```

```go
type Tool interface {
    Name() string
    Description() string
    Schema() map[string]any
    Execute(ctx context.Context, call ToolCall, update ToolUpdateFunc) (ToolResult, error)
}
```

Events should include:

```go
AgentStart
TurnStart
MessageStart
MessageDelta
MessageEnd
ToolStart
ToolUpdate
ToolEnd
TurnEnd
AgentEnd
```

## Implementation Plan

1. Scaffold the Go project with `go.mod`, package directories, and a small example CLI.
2. Implement provider-neutral message types:
   - `UserMessage`
   - `AssistantMessage`
   - `ToolResultMessage`
   - content blocks for text, image placeholder, thinking placeholder, and tool calls.
3. Implement the low-level agent loop:
   - append user prompt
   - call provider stream
   - emit assistant deltas
   - finalize assistant message
   - execute requested tools
   - append tool results
   - continue until no tool calls remain.
4. Implement the stateful `Agent` wrapper:
   - owns transcript, tools, model config, active run, and subscribers
   - prevents concurrent prompts
   - supports `Steer` and `FollowUp` queues
   - uses `context.Context` cancellation for aborts.
5. Implement OpenRouter provider:
   - read `OPENROUTER_API_KEY`
   - send current system prompt, messages, and tools
   - stream text deltas and tool calls
   - return provider errors as assistant error events where possible.
6. Implement tool execution:
   - default parallel execution for multiple tool calls
   - sequential execution option can be added after v1 loop works
   - errors become tool result messages with `IsError: true`.
7. Add JSONL session persistence after in-memory behavior is stable.

## Test Plan

- Unit tests with a fake provider:
  - simple prompt emits lifecycle events in order
  - assistant text streaming updates state
  - one tool call executes and feeds result back to model
  - multiple tool calls execute without losing source order
  - tool error becomes an error tool result
  - abort cancels provider stream and tool execution
  - `Steer` injects after current turn
  - `FollowUp` runs after the agent would otherwise stop.
- OpenRouter integration smoke test:
  - skipped unless `OPENROUTER_API_KEY` is set
  - prompt returns streamed text
  - tool call round trip works with one simple calculator tool.
- Example CLI smoke:
  - `go run ./examples/simple "Say hi"`
  - `go run ./examples/tool-call "what is 2+3?"`

## Assumptions

- The selected file path is `/Users/nishanthgowda/Developer/go-agent-core/PLAN.md`.
- v1 uses OpenRouter only.
- The goal is to mimic agent-core capabilities in idiomatic Go, not preserve npm imports or TypeScript declaration-merging behavior.
- Persistence, compaction, skills, and multi-provider support are explicitly post-v1.
