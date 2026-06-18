package tool

import (
	"context"
	"encoding/json"

	"go-agent-core/llm"
)

// Tool describes executable functionality that can be exposed to a model.
type Tool interface {
	Name() string
	Description() string
	Schema() map[string]any
	Execute(ctx context.Context, call Call, update UpdateFunc) (Result, error)
}

// Call is a normalized assistant tool request.
type Call struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// Result is the normalized output from a tool execution.
type Result struct {
	CallID  string
	Name    string
	IsError bool
	Content []llm.ContentBlock
}

// Update describes incremental tool progress.
type Update struct {
	Message string
	Data    any
}

// UpdateFunc receives progress updates from a running tool.
type UpdateFunc func(Update)

// Specs converts executable tools into provider-neutral tool metadata.
func Specs(tools []Tool) []llm.ToolSpec {
	specs := make([]llm.ToolSpec, 0, len(tools))
	for _, tool := range tools {
		specs = append(specs, llm.ToolSpec{
			Name:        tool.Name(),
			Description: tool.Description(),
			Schema:      tool.Schema(),
		})
	}
	return specs
}
