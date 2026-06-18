package llm

import (
	"encoding/json"
	"testing"
)

func TestMessageRolesAndContent(t *testing.T) {
	messages := []Message{
		UserMessage{Content: []ContentBlock{TextBlock{Text: "hello"}}},
		AssistantMessage{Content: []ContentBlock{ThinkingBlock{Text: "checking"}, TextBlock{Text: "hi"}}},
		ToolResultMessage{
			ToolCallID: "call_1",
			ToolName:   "calculator",
			Content:    []ContentBlock{TextBlock{Text: "5"}},
		},
	}

	wantRoles := []Role{RoleUser, RoleAssistant, RoleTool}
	for i, msg := range messages {
		if got := msg.MessageRole(); got != wantRoles[i] {
			t.Fatalf("message %d role = %q, want %q", i, got, wantRoles[i])
		}
		if len(msg.MessageContent()) == 0 {
			t.Fatalf("message %d content is empty", i)
		}
	}
}

func TestContentBlockKinds(t *testing.T) {
	blocks := []ContentBlock{
		TextBlock{Text: "hello"},
		ImageBlock{URL: "file://image.png", MediaType: "image/png"},
		ThinkingBlock{Text: "thinking"},
		ToolCallBlock{ID: "call_1", Name: "calculator", Arguments: json.RawMessage(`{"a":2,"b":3}`)},
	}

	wantKinds := []ContentKind{
		ContentKindText,
		ContentKindImage,
		ContentKindThinking,
		ContentKindToolCall,
	}

	for i, block := range blocks {
		if got := block.ContentKind(); got != wantKinds[i] {
			t.Fatalf("block %d kind = %q, want %q", i, got, wantKinds[i])
		}
	}
}
