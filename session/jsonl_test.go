package session

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"go-agent-core/llm"
)

func TestJSONLRoundTrip(t *testing.T) {
	messages := []llm.Message{
		llm.UserMessage{Content: []llm.ContentBlock{
			llm.TextBlock{Text: "hello"},
			llm.ImageBlock{URL: "https://example.test/image.png", MediaType: "image/png"},
		}},
		llm.AssistantMessage{Content: []llm.ContentBlock{
			llm.ThinkingBlock{Text: "checking"},
			llm.TextBlock{Text: "calling"},
			llm.ToolCallBlock{ID: "call_1", Name: "calculator", Arguments: json.RawMessage(`{"a":2,"b":3}`)},
		}},
		llm.ToolResultMessage{
			ToolCallID: "call_1",
			ToolName:   "calculator",
			Content:    []llm.ContentBlock{llm.TextBlock{Text: "5"}},
		},
		llm.ToolResultMessage{
			ToolCallID: "call_2",
			ToolName:   "broken",
			IsError:    true,
			Content:    []llm.ContentBlock{llm.TextBlock{Text: "boom"}},
		},
	}

	var buf bytes.Buffer
	if err := WriteJSONL(&buf, messages); err != nil {
		t.Fatalf("WriteJSONL returned error: %v", err)
	}

	got, err := ReadJSONL(&buf)
	if err != nil {
		t.Fatalf("ReadJSONL returned error: %v", err)
	}
	if !reflect.DeepEqual(got, messages) {
		t.Fatalf("messages = %#v, want %#v", got, messages)
	}
}

func TestSaveAndLoadJSONL(t *testing.T) {
	path := t.TempDir() + "/session.jsonl"
	want := []llm.Message{
		llm.UserMessage{Content: []llm.ContentBlock{llm.TextBlock{Text: "persist me"}}},
		llm.AssistantMessage{Content: []llm.ContentBlock{llm.TextBlock{Text: "loaded"}}},
	}

	if err := SaveJSONL(path, want); err != nil {
		t.Fatalf("SaveJSONL returned error: %v", err)
	}
	got, err := LoadJSONL(path)
	if err != nil {
		t.Fatalf("LoadJSONL returned error: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("messages = %#v, want %#v", got, want)
	}
}

func TestReadJSONLReportsMalformedLine(t *testing.T) {
	_, err := ReadJSONL(strings.NewReader("{\"role\":\"user\",\"content\":[]}\nnot-json\n"))
	if err == nil {
		t.Fatal("ReadJSONL returned nil error")
	}
	if !strings.Contains(err.Error(), "line 2") {
		t.Fatalf("error = %q, want line number", err.Error())
	}
}

func TestReadJSONLReportsUnknownRole(t *testing.T) {
	_, err := ReadJSONL(strings.NewReader("{\"role\":\"alien\",\"content\":[]}\n"))
	if err == nil {
		t.Fatal("ReadJSONL returned nil error")
	}
	if !strings.Contains(err.Error(), "unknown message role") {
		t.Fatalf("error = %q, want unknown role", err.Error())
	}
}

func TestReadJSONLReportsUnknownContentKind(t *testing.T) {
	_, err := ReadJSONL(strings.NewReader("{\"role\":\"user\",\"content\":[{\"kind\":\"audio\"}]}\n"))
	if err == nil {
		t.Fatal("ReadJSONL returned nil error")
	}
	if !strings.Contains(err.Error(), "unknown content kind") {
		t.Fatalf("error = %q, want unknown content kind", err.Error())
	}
}

func TestWriteJSONLReportsUnsupportedMessage(t *testing.T) {
	var buf bytes.Buffer
	err := WriteJSONL(&buf, []llm.Message{unsupportedMessage{}})
	if err == nil {
		t.Fatal("WriteJSONL returned nil error")
	}
	if !strings.Contains(err.Error(), "unsupported message type") {
		t.Fatalf("error = %q, want unsupported message type", err.Error())
	}
}

func TestWriteJSONLReportsUnsupportedContentBlock(t *testing.T) {
	var buf bytes.Buffer
	err := WriteJSONL(&buf, []llm.Message{
		llm.UserMessage{Content: []llm.ContentBlock{unsupportedBlock{}}},
	})
	if err == nil {
		t.Fatal("WriteJSONL returned nil error")
	}
	if !strings.Contains(err.Error(), "unsupported content block type") {
		t.Fatalf("error = %q, want unsupported content block type", err.Error())
	}
}

type unsupportedMessage struct{}

func (unsupportedMessage) MessageRole() llm.Role {
	return llm.RoleUser
}

func (unsupportedMessage) MessageContent() []llm.ContentBlock {
	return nil
}

type unsupportedBlock struct{}

func (unsupportedBlock) ContentKind() llm.ContentKind {
	return "unsupported"
}

func TestReadJSONLReturnsScannerError(t *testing.T) {
	_, err := ReadJSONL(errorReader{})
	if err == nil {
		t.Fatal("ReadJSONL returned nil error")
	}
	if !strings.Contains(err.Error(), "read JSONL") {
		t.Fatalf("error = %q, want read JSONL", err.Error())
	}
}

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) {
	return 0, errors.New("read failed")
}
