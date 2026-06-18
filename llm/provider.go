package llm

import (
	"context"
	"encoding/json"
)

// Provider streams model output for one request.
type Provider interface {
	Stream(ctx context.Context, req Request) (<-chan ProviderEvent, error)
}

// Request is the provider-neutral input for one model turn.
type Request struct {
	SystemPrompt string
	Messages     []Message
	Tools        []ToolSpec
}

// ToolSpec describes a tool available to the model.
type ToolSpec struct {
	Name        string
	Description string
	Schema      map[string]any
}

// ProviderEventKind identifies a streamed provider event.
type ProviderEventKind string

const (
	ProviderEventTextDelta ProviderEventKind = "text_delta"
	ProviderEventToolCall  ProviderEventKind = "tool_call"
	ProviderEventError     ProviderEventKind = "error"
)

// ProviderEvent is one streamed model event.
type ProviderEvent interface {
	ProviderEventKind() ProviderEventKind
}

// TextDeltaEvent contains a streamed assistant text delta.
type TextDeltaEvent struct {
	Delta string
}

func (e TextDeltaEvent) ProviderEventKind() ProviderEventKind {
	return ProviderEventTextDelta
}

// ToolCallEvent contains one complete streamed tool call.
type ToolCallEvent struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

func (e ToolCallEvent) ProviderEventKind() ProviderEventKind {
	return ProviderEventToolCall
}

// ErrorEvent reports a provider error that occurs after streaming starts.
type ErrorEvent struct {
	Err error
}

func (e ErrorEvent) ProviderEventKind() ProviderEventKind {
	return ProviderEventError
}
