package openrouter

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"go-agent-core/llm"
)

func TestNewDefaultsToDeepSeekV4Flash(t *testing.T) {
	provider, err := New(Config{APIKey: "test-key"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if provider.model != "deepseek/deepseek-v4-flash" {
		t.Fatalf("model = %q, want deepseek/deepseek-v4-flash", provider.model)
	}
}

func TestStreamEncodesChatCompletionRequest(t *testing.T) {
	var got chatRequest
	var auth string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("path = %q, want /chat/completions", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: server.URL,
		Model:   "anthropic/claude-sonnet-4.5",
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	stream, err := provider.Stream(context.Background(), llm.Request{
		SystemPrompt: "answer clearly",
		Messages: []llm.Message{
			llm.UserMessage{Content: []llm.ContentBlock{llm.TextBlock{Text: "hello"}}},
			llm.AssistantMessage{Content: []llm.ContentBlock{
				llm.TextBlock{Text: "calling"},
				llm.ToolCallBlock{ID: "call_1", Name: "calculator", Arguments: json.RawMessage(`{"a":2,"b":3}`)},
			}},
			llm.ToolResultMessage{
				ToolCallID: "call_1",
				ToolName:   "calculator",
				Content:    []llm.ContentBlock{llm.TextBlock{Text: "5"}},
			},
		},
		Tools: []llm.ToolSpec{{
			Name:        "calculator",
			Description: "Adds two numbers",
			Schema:      map[string]any{"type": "object"},
		}},
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	for range stream {
	}

	if auth != "Bearer test-key" {
		t.Fatalf("authorization = %q, want Bearer test-key", auth)
	}
	if got.Model != "anthropic/claude-sonnet-4.5" {
		t.Fatalf("model = %q", got.Model)
	}
	if !got.Stream {
		t.Fatal("stream = false, want true")
	}

	wantRoles := []string{"system", "user", "assistant", "tool"}
	var gotRoles []string
	for _, msg := range got.Messages {
		gotRoles = append(gotRoles, msg.Role)
	}
	if !reflect.DeepEqual(gotRoles, wantRoles) {
		t.Fatalf("roles = %#v, want %#v", gotRoles, wantRoles)
	}
	if got.Messages[2].ToolCalls[0].Function.Arguments != `{"a":2,"b":3}` {
		t.Fatalf("tool call args = %q", got.Messages[2].ToolCalls[0].Function.Arguments)
	}
	if got.Messages[3].ToolCallID != "call_1" {
		t.Fatalf("tool result call id = %q", got.Messages[3].ToolCallID)
	}
	if got.Tools[0].Type != "function" || got.Tools[0].Function.Name != "calculator" {
		t.Fatalf("tool encoding = %#v", got.Tools[0])
	}
}

func TestStreamParsesTextDeltas(t *testing.T) {
	server := httptest.NewServer(sseHandler([]string{
		`{"choices":[{"delta":{"content":"hel"}}]}`,
		`: OPENROUTER PROCESSING`,
		`{"choices":[{"delta":{"content":"lo"}}]}`,
		`[DONE]`,
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	events := collectEvents(t, provider)

	want := []llm.ProviderEvent{
		llm.TextDeltaEvent{Delta: "hel"},
		llm.TextDeltaEvent{Delta: "lo"},
	}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %#v, want %#v", events, want)
	}
}

func TestStreamParsesToolCallDeltas(t *testing.T) {
	server := httptest.NewServer(sseHandler([]string{
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"calculator","arguments":"{\"a\":"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"2,\"b\":3}"}}]},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	events := collectEvents(t, provider)

	want := []llm.ProviderEvent{
		llm.ToolCallEvent{ID: "call_1", Name: "calculator", Arguments: json.RawMessage(`{"a":2,"b":3}`)},
	}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %#v, want %#v", events, want)
	}
}

func TestStreamReturnsHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key"}}`))
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	_, err := provider.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.UserMessage{Content: []llm.ContentBlock{llm.TextBlock{Text: "hello"}}}},
	})
	if err == nil {
		t.Fatal("Stream returned nil error, want HTTP error")
	}
}

func TestStreamParsesMidStreamError(t *testing.T) {
	server := httptest.NewServer(sseHandler([]string{
		`{"error":{"message":"provider disconnected"},"choices":[{"finish_reason":"error"}]}`,
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	events := collectEvents(t, provider)

	if len(events) != 1 {
		t.Fatalf("event count = %d, want 1", len(events))
	}
	event, ok := events[0].(llm.ErrorEvent)
	if !ok {
		t.Fatalf("event type = %T, want llm.ErrorEvent", events[0])
	}
	if event.Err == nil || event.Err.Error() != "provider disconnected" {
		t.Fatalf("error = %v, want provider disconnected", event.Err)
	}
}

func newTestProvider(t *testing.T, baseURL string) *Provider {
	t.Helper()

	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: baseURL,
		Model:   "test/model",
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return provider
}

func collectEvents(t *testing.T, provider *Provider) []llm.ProviderEvent {
	t.Helper()

	stream, err := provider.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{llm.UserMessage{Content: []llm.ContentBlock{llm.TextBlock{Text: "hello"}}}},
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	var events []llm.ProviderEvent
	for event := range stream {
		events = append(events, event)
	}
	return events
}

func sseHandler(lines []string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range lines {
			if len(line) > 0 && line[0] == ':' {
				_, _ = w.Write([]byte(line + "\n\n"))
				continue
			}
			_, _ = w.Write([]byte("data: " + line + "\n\n"))
		}
	}
}
