// Package session persists provider-neutral transcripts.
package session

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"go-agent-core/llm"
)

type messageRecord struct {
	Role       llm.Role        `json:"role"`
	Content    []contentRecord `json:"content"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolName   string          `json:"tool_name,omitempty"`
	IsError    bool            `json:"is_error,omitempty"`
}

type contentRecord struct {
	Kind      llm.ContentKind `json:"kind"`
	Text      string          `json:"text,omitempty"`
	URL       string          `json:"url,omitempty"`
	MediaType string          `json:"media_type,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// SaveJSONL writes messages to path as newline-delimited JSON, replacing any existing file.
func SaveJSONL(path string, messages []llm.Message) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return WriteJSONL(file, messages)
}

// LoadJSONL reads newline-delimited JSON messages from path.
func LoadJSONL(path string) ([]llm.Message, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	return ReadJSONL(file)
}

// WriteJSONL writes messages as newline-delimited JSON.
func WriteJSONL(w io.Writer, messages []llm.Message) error {
	encoder := json.NewEncoder(w)
	for i, message := range messages {
		record, err := encodeMessage(message)
		if err != nil {
			return fmt.Errorf("session: encode message %d: %w", i+1, err)
		}
		if err := encoder.Encode(record); err != nil {
			return fmt.Errorf("session: write message %d: %w", i+1, err)
		}
	}
	return nil
}

// ReadJSONL reads newline-delimited JSON into provider-neutral messages.
func ReadJSONL(r io.Reader) ([]llm.Message, error) {
	scanner := bufio.NewScanner(r)
	var messages []llm.Message
	for line := 1; scanner.Scan(); line++ {
		bytes := scanner.Bytes()
		if len(bytes) == 0 {
			continue
		}

		var record messageRecord
		if err := json.Unmarshal(bytes, &record); err != nil {
			return nil, fmt.Errorf("session: decode line %d: %w", line, err)
		}

		message, err := decodeMessage(record)
		if err != nil {
			return nil, fmt.Errorf("session: decode line %d: %w", line, err)
		}
		messages = append(messages, message)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("session: read JSONL: %w", err)
	}
	return messages, nil
}

func encodeMessage(message llm.Message) (messageRecord, error) {
	switch message := message.(type) {
	case llm.UserMessage:
		content, err := encodeContent(message.Content)
		return messageRecord{Role: llm.RoleUser, Content: content}, err
	case llm.AssistantMessage:
		content, err := encodeContent(message.Content)
		return messageRecord{Role: llm.RoleAssistant, Content: content}, err
	case llm.ToolResultMessage:
		content, err := encodeContent(message.Content)
		return messageRecord{
			Role:       llm.RoleTool,
			Content:    content,
			ToolCallID: message.ToolCallID,
			ToolName:   message.ToolName,
			IsError:    message.IsError,
		}, err
	default:
		return messageRecord{}, fmt.Errorf("unsupported message type %T", message)
	}
}

func decodeMessage(record messageRecord) (llm.Message, error) {
	content, err := decodeContent(record.Content)
	if err != nil {
		return nil, err
	}

	switch record.Role {
	case llm.RoleUser:
		return llm.UserMessage{Content: content}, nil
	case llm.RoleAssistant:
		return llm.AssistantMessage{Content: content}, nil
	case llm.RoleTool:
		return llm.ToolResultMessage{
			ToolCallID: record.ToolCallID,
			ToolName:   record.ToolName,
			IsError:    record.IsError,
			Content:    content,
		}, nil
	default:
		return nil, fmt.Errorf("unknown message role %q", record.Role)
	}
}

func encodeContent(blocks []llm.ContentBlock) ([]contentRecord, error) {
	records := make([]contentRecord, 0, len(blocks))
	for i, block := range blocks {
		record, err := encodeContentBlock(block)
		if err != nil {
			return nil, fmt.Errorf("content block %d: %w", i+1, err)
		}
		records = append(records, record)
	}
	return records, nil
}

func decodeContent(records []contentRecord) ([]llm.ContentBlock, error) {
	blocks := make([]llm.ContentBlock, 0, len(records))
	for i, record := range records {
		block, err := decodeContentBlock(record)
		if err != nil {
			return nil, fmt.Errorf("content block %d: %w", i+1, err)
		}
		blocks = append(blocks, block)
	}
	return blocks, nil
}

func encodeContentBlock(block llm.ContentBlock) (contentRecord, error) {
	switch block := block.(type) {
	case llm.TextBlock:
		return contentRecord{Kind: llm.ContentKindText, Text: block.Text}, nil
	case llm.ImageBlock:
		return contentRecord{Kind: llm.ContentKindImage, URL: block.URL, MediaType: block.MediaType}, nil
	case llm.ThinkingBlock:
		return contentRecord{Kind: llm.ContentKindThinking, Text: block.Text}, nil
	case llm.ToolCallBlock:
		return contentRecord{
			Kind:      llm.ContentKindToolCall,
			ID:        block.ID,
			Name:      block.Name,
			Arguments: block.Arguments,
		}, nil
	default:
		return contentRecord{}, fmt.Errorf("unsupported content block type %T", block)
	}
}

func decodeContentBlock(record contentRecord) (llm.ContentBlock, error) {
	switch record.Kind {
	case llm.ContentKindText:
		return llm.TextBlock{Text: record.Text}, nil
	case llm.ContentKindImage:
		return llm.ImageBlock{URL: record.URL, MediaType: record.MediaType}, nil
	case llm.ContentKindThinking:
		return llm.ThinkingBlock{Text: record.Text}, nil
	case llm.ContentKindToolCall:
		return llm.ToolCallBlock{
			ID:        record.ID,
			Name:      record.Name,
			Arguments: append(json.RawMessage(nil), record.Arguments...),
		}, nil
	default:
		return nil, fmt.Errorf("unknown content kind %q", record.Kind)
	}
}
