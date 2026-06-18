package llm

import "encoding/json"

// Role identifies who produced a message in the transcript.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is a provider-neutral transcript entry.
type Message interface {
	MessageRole() Role
	MessageContent() []ContentBlock
}

// UserMessage contains content supplied by the user.
type UserMessage struct {
	Content []ContentBlock
}

func (m UserMessage) MessageRole() Role {
	return RoleUser
}

func (m UserMessage) MessageContent() []ContentBlock {
	return m.Content
}

// AssistantMessage contains content produced by the model.
type AssistantMessage struct {
	Content []ContentBlock
}

func (m AssistantMessage) MessageRole() Role {
	return RoleAssistant
}

func (m AssistantMessage) MessageContent() []ContentBlock {
	return m.Content
}

// ToolResultMessage contains the result returned for one assistant tool call.
type ToolResultMessage struct {
	ToolCallID string
	ToolName   string
	IsError    bool
	Content    []ContentBlock
}

func (m ToolResultMessage) MessageRole() Role {
	return RoleTool
}

func (m ToolResultMessage) MessageContent() []ContentBlock {
	return m.Content
}

// ContentKind identifies the shape of a content block.
type ContentKind string

const (
	ContentKindText     ContentKind = "text"
	ContentKindImage    ContentKind = "image"
	ContentKindThinking ContentKind = "thinking"
	ContentKindToolCall ContentKind = "tool_call"
)

// ContentBlock is one typed part of a message.
type ContentBlock interface {
	ContentKind() ContentKind
}

// TextBlock contains plain text content.
type TextBlock struct {
	Text string
}

func (b TextBlock) ContentKind() ContentKind {
	return ContentKindText
}

// ImageBlock is a provider-neutral placeholder for image input.
type ImageBlock struct {
	URL       string
	MediaType string
}

func (b ImageBlock) ContentKind() ContentKind {
	return ContentKindImage
}

// ThinkingBlock is a provider-neutral placeholder for model reasoning content.
type ThinkingBlock struct {
	Text string
}

func (b ThinkingBlock) ContentKind() ContentKind {
	return ContentKindThinking
}

// ToolCallBlock requests execution of a named tool with JSON arguments.
type ToolCallBlock struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

func (b ToolCallBlock) ContentKind() ContentKind {
	return ContentKindToolCall
}
