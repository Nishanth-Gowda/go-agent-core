package agent

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"go-agent-core/llm"
	"go-agent-core/tool"
)

func TestRunLoopStreamsAssistantMessage(t *testing.T) {
	provider := &fakeProvider{
		responses: [][]llm.ProviderEvent{{
			llm.TextDeltaEvent{Delta: "hi"},
			llm.TextDeltaEvent{Delta: " there"},
		}},
	}
	var events []EventKind

	messages, err := RunLoop(context.Background(), nil, "hello", LoopOptions{
		Provider: provider,
		OnEvent: func(event Event) {
			events = append(events, event.Kind)
		},
	})
	if err != nil {
		t.Fatalf("RunLoop returned error: %v", err)
	}

	if len(messages) != 2 {
		t.Fatalf("message count = %d, want 2", len(messages))
	}
	assertTextMessage(t, messages[0], llm.RoleUser, "hello")
	assertTextMessage(t, messages[1], llm.RoleAssistant, "hi there")

	wantEvents := []EventKind{
		EventAgentStart,
		EventTurnStart,
		EventMessageStart,
		EventMessageDelta,
		EventMessageDelta,
		EventMessageEnd,
		EventTurnEnd,
		EventAgentEnd,
	}
	if !reflect.DeepEqual(events, wantEvents) {
		t.Fatalf("events = %#v, want %#v", events, wantEvents)
	}
}

func TestRunLoopExecutesToolAndContinues(t *testing.T) {
	provider := &fakeProvider{
		responses: [][]llm.ProviderEvent{
			{
				llm.ToolCallEvent{ID: "call_1", Name: "calculator", Arguments: json.RawMessage(`{"a":2,"b":3}`)},
			},
			{
				llm.TextDeltaEvent{Delta: "result is 5"},
			},
		},
	}

	messages, err := RunLoop(context.Background(), nil, "what is 2+3?", LoopOptions{
		Provider: provider,
		Tools: []tool.Tool{
			fakeTool{name: "calculator", content: []llm.ContentBlock{llm.TextBlock{Text: "5"}}},
		},
	})
	if err != nil {
		t.Fatalf("RunLoop returned error: %v", err)
	}

	if len(provider.requests) != 2 {
		t.Fatalf("provider calls = %d, want 2", len(provider.requests))
	}
	if len(messages) != 4 {
		t.Fatalf("message count = %d, want 4", len(messages))
	}

	assistant, ok := messages[1].(llm.AssistantMessage)
	if !ok {
		t.Fatalf("message 1 type = %T, want llm.AssistantMessage", messages[1])
	}
	if got := assistant.Content[0].(llm.ToolCallBlock).Name; got != "calculator" {
		t.Fatalf("tool call name = %q, want calculator", got)
	}
	assertToolResult(t, messages[2], "call_1", "calculator", false, "5")
	assertTextMessage(t, messages[3], llm.RoleAssistant, "result is 5")

	if len(provider.requests[1].Messages) != 3 {
		t.Fatalf("second provider request message count = %d, want 3", len(provider.requests[1].Messages))
	}
	assertToolResult(t, provider.requests[1].Messages[2], "call_1", "calculator", false, "5")
}

func TestRunLoopPreservesMultipleToolResultOrder(t *testing.T) {
	provider := &fakeProvider{
		responses: [][]llm.ProviderEvent{
			{
				llm.ToolCallEvent{ID: "call_b", Name: "second"},
				llm.ToolCallEvent{ID: "call_a", Name: "first"},
			},
			{
				llm.TextDeltaEvent{Delta: "done"},
			},
		},
	}

	messages, err := RunLoop(context.Background(), nil, "run tools", LoopOptions{
		Provider: provider,
		Tools: []tool.Tool{
			fakeTool{name: "first", content: []llm.ContentBlock{llm.TextBlock{Text: "first result"}}},
			fakeTool{name: "second", content: []llm.ContentBlock{llm.TextBlock{Text: "second result"}}},
		},
	})
	if err != nil {
		t.Fatalf("RunLoop returned error: %v", err)
	}

	assertToolResult(t, messages[2], "call_b", "second", false, "second result")
	assertToolResult(t, messages[3], "call_a", "first", false, "first result")
}

func TestRunLoopConvertsToolErrorToToolResult(t *testing.T) {
	provider := &fakeProvider{
		responses: [][]llm.ProviderEvent{
			{
				llm.ToolCallEvent{ID: "call_1", Name: "broken"},
			},
			{
				llm.TextDeltaEvent{Delta: "handled"},
			},
		},
	}

	messages, err := RunLoop(context.Background(), nil, "run broken tool", LoopOptions{
		Provider: provider,
		Tools: []tool.Tool{
			fakeTool{name: "broken", err: errors.New("boom")},
		},
	})
	if err != nil {
		t.Fatalf("RunLoop returned error: %v", err)
	}

	assertToolResult(t, messages[2], "call_1", "broken", true, "boom")
	assertToolResult(t, provider.requests[1].Messages[2], "call_1", "broken", true, "boom")
}

type fakeProvider struct {
	responses [][]llm.ProviderEvent
	requests  []llm.Request
}

func (p *fakeProvider) Stream(ctx context.Context, req llm.Request) (<-chan llm.ProviderEvent, error) {
	p.requests = append(p.requests, req)
	call := len(p.requests) - 1
	if call >= len(p.responses) {
		return nil, errors.New("unexpected provider call")
	}

	events := p.responses[call]
	stream := make(chan llm.ProviderEvent, len(events))
	for _, event := range events {
		stream <- event
	}
	close(stream)

	return stream, nil
}

type fakeTool struct {
	name    string
	content []llm.ContentBlock
	err     error
}

func (t fakeTool) Name() string {
	return t.name
}

func (t fakeTool) Description() string {
	return "fake tool"
}

func (t fakeTool) Schema() map[string]any {
	return nil
}

func (t fakeTool) Execute(ctx context.Context, call tool.Call, update tool.UpdateFunc) (tool.Result, error) {
	if t.err != nil {
		return tool.Result{}, t.err
	}
	return tool.Result{
		CallID:  call.ID,
		Name:    call.Name,
		Content: t.content,
	}, nil
}

func assertTextMessage(t *testing.T, message llm.Message, role llm.Role, text string) {
	t.Helper()

	if got := message.MessageRole(); got != role {
		t.Fatalf("message role = %q, want %q", got, role)
	}

	content := message.MessageContent()
	if len(content) != 1 {
		t.Fatalf("content block count = %d, want 1", len(content))
	}

	block, ok := content[0].(llm.TextBlock)
	if !ok {
		t.Fatalf("content block type = %T, want llm.TextBlock", content[0])
	}
	if block.Text != text {
		t.Fatalf("text = %q, want %q", block.Text, text)
	}
}

func assertToolResult(t *testing.T, message llm.Message, callID string, name string, isError bool, text string) {
	t.Helper()

	result, ok := message.(llm.ToolResultMessage)
	if !ok {
		t.Fatalf("message type = %T, want llm.ToolResultMessage", message)
	}
	if result.ToolCallID != callID {
		t.Fatalf("tool call id = %q, want %q", result.ToolCallID, callID)
	}
	if result.ToolName != name {
		t.Fatalf("tool name = %q, want %q", result.ToolName, name)
	}
	if result.IsError != isError {
		t.Fatalf("is error = %v, want %v", result.IsError, isError)
	}
	assertTextMessage(t, result, llm.RoleTool, text)
}
