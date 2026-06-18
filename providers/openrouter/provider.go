package openrouter

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"go-agent-core/llm"
)

const (
	defaultBaseURL = "https://openrouter.ai/api/v1"
	defaultModel   = "~openai/gpt-latest"
)

// Config configures an OpenRouter provider.
type Config struct {
	APIKey      string
	BaseURL     string
	Model       string
	HTTPClient  *http.Client
	HTTPReferer string
	AppTitle    string
}

// Provider streams model output from OpenRouter's chat completions API.
type Provider struct {
	apiKey      string
	baseURL     string
	model       string
	httpClient  *http.Client
	httpReferer string
	appTitle    string
}

// New creates an OpenRouter provider. If APIKey is empty, OPENROUTER_API_KEY is used.
func New(config Config) (*Provider, error) {
	apiKey := strings.TrimSpace(config.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	}
	if apiKey == "" {
		return nil, errors.New("openrouter: API key is required")
	}

	baseURL := strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	model := strings.TrimSpace(config.Model)
	if model == "" {
		model = defaultModel
	}

	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	return &Provider{
		apiKey:      apiKey,
		baseURL:     baseURL,
		model:       model,
		httpClient:  httpClient,
		httpReferer: config.HTTPReferer,
		appTitle:    config.AppTitle,
	}, nil
}

// Stream sends one streaming chat-completion request and returns provider-neutral events.
func (p *Provider) Stream(ctx context.Context, req llm.Request) (<-chan llm.ProviderEvent, error) {
	if p == nil {
		return nil, errors.New("openrouter: provider is nil")
	}

	body, err := p.requestBody(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	if p.httpReferer != "" {
		httpReq.Header.Set("HTTP-Referer", p.httpReferer)
	}
	if p.appTitle != "" {
		httpReq.Header.Set("X-OpenRouter-Title", p.appTitle)
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		return nil, readErrorResponse(resp)
	}

	events := make(chan llm.ProviderEvent)
	go func() {
		defer close(events)
		defer resp.Body.Close()
		streamSSE(ctx, resp.Body, events)
	}()

	return events, nil
}

func (p *Provider) requestBody(req llm.Request) ([]byte, error) {
	messages, err := encodeMessages(req)
	if err != nil {
		return nil, err
	}

	body := chatRequest{
		Model:    p.model,
		Messages: messages,
		Stream:   true,
	}
	if len(req.Tools) > 0 {
		body.Tools = encodeTools(req.Tools)
	}

	return json.Marshal(body)
}

func encodeMessages(req llm.Request) ([]chatMessage, error) {
	messages := make([]chatMessage, 0, len(req.Messages)+1)
	if strings.TrimSpace(req.SystemPrompt) != "" {
		messages = append(messages, chatMessage{
			Role:    "system",
			Content: req.SystemPrompt,
		})
	}

	for _, msg := range req.Messages {
		encoded, err := encodeMessage(msg)
		if err != nil {
			return nil, err
		}
		messages = append(messages, encoded)
	}

	return messages, nil
}

func encodeMessage(msg llm.Message) (chatMessage, error) {
	switch msg := msg.(type) {
	case llm.UserMessage:
		content, err := textContent(msg.Content)
		if err != nil {
			return chatMessage{}, err
		}
		return chatMessage{Role: "user", Content: content}, nil
	case llm.AssistantMessage:
		content, toolCalls, err := assistantContent(msg.Content)
		if err != nil {
			return chatMessage{}, err
		}
		return chatMessage{Role: "assistant", Content: content, ToolCalls: toolCalls}, nil
	case llm.ToolResultMessage:
		content, err := textContent(msg.Content)
		if err != nil {
			return chatMessage{}, err
		}
		return chatMessage{
			Role:       "tool",
			Content:    content,
			ToolCallID: msg.ToolCallID,
			Name:       msg.ToolName,
		}, nil
	default:
		return chatMessage{}, fmt.Errorf("openrouter: unsupported message type %T", msg)
	}
}

func textContent(blocks []llm.ContentBlock) (string, error) {
	var out strings.Builder
	for _, block := range blocks {
		switch block := block.(type) {
		case llm.TextBlock:
			out.WriteString(block.Text)
		case llm.ThinkingBlock:
			continue
		default:
			return "", fmt.Errorf("openrouter: unsupported content block %T in text message", block)
		}
	}
	return out.String(), nil
}

func assistantContent(blocks []llm.ContentBlock) (string, []chatToolCall, error) {
	var out strings.Builder
	var calls []chatToolCall
	for _, block := range blocks {
		switch block := block.(type) {
		case llm.TextBlock:
			out.WriteString(block.Text)
		case llm.ThinkingBlock:
			continue
		case llm.ToolCallBlock:
			calls = append(calls, chatToolCall{
				ID:   block.ID,
				Type: "function",
				Function: chatToolCallFunction{
					Name:      block.Name,
					Arguments: string(block.Arguments),
				},
			})
		default:
			return "", nil, fmt.Errorf("openrouter: unsupported assistant content block %T", block)
		}
	}
	return out.String(), calls, nil
}

func encodeTools(specs []llm.ToolSpec) []chatTool {
	tools := make([]chatTool, 0, len(specs))
	for _, spec := range specs {
		tools = append(tools, chatTool{
			Type: "function",
			Function: chatToolFunction{
				Name:        spec.Name,
				Description: spec.Description,
				Parameters:  spec.Schema,
			},
		})
	}
	return tools
}

func streamSSE(ctx context.Context, body io.Reader, events chan<- llm.ProviderEvent) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	calls := make(map[int]*toolCallAccumulator)
	var order []int
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			sendEvent(ctx, events, llm.ErrorEvent{Err: ctx.Err()})
			return
		default:
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			flushToolCalls(ctx, events, calls, order)
			return
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			sendEvent(ctx, events, llm.ErrorEvent{Err: err})
			return
		}
		if chunk.Error != nil {
			sendEvent(ctx, events, llm.ErrorEvent{Err: errors.New(chunk.Error.Message)})
			return
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				if !sendEvent(ctx, events, llm.TextDeltaEvent{Delta: choice.Delta.Content}) {
					return
				}
			}
			for _, call := range choice.Delta.ToolCalls {
				if _, ok := calls[call.Index]; !ok {
					calls[call.Index] = &toolCallAccumulator{}
					order = append(order, call.Index)
				}
				acc := calls[call.Index]
				if call.ID != "" {
					acc.id = call.ID
				}
				if call.Function.Name != "" {
					acc.name = call.Function.Name
				}
				acc.arguments.WriteString(call.Function.Arguments)
			}
			if choice.FinishReason == "tool_calls" {
				flushToolCalls(ctx, events, calls, order)
				calls = make(map[int]*toolCallAccumulator)
				order = nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		sendEvent(ctx, events, llm.ErrorEvent{Err: err})
		return
	}
	flushToolCalls(ctx, events, calls, order)
}

func flushToolCalls(ctx context.Context, events chan<- llm.ProviderEvent, calls map[int]*toolCallAccumulator, order []int) {
	for _, index := range order {
		call := calls[index]
		if call == nil || call.name == "" {
			continue
		}
		if !sendEvent(ctx, events, llm.ToolCallEvent{
			ID:        call.id,
			Name:      call.name,
			Arguments: json.RawMessage(call.arguments.String()),
		}) {
			return
		}
	}
}

func sendEvent(ctx context.Context, events chan<- llm.ProviderEvent, event llm.ProviderEvent) bool {
	select {
	case <-ctx.Done():
		return false
	case events <- event:
		return true
	}
}

func readErrorResponse(resp *http.Response) error {
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("openrouter: request failed with status %d", resp.StatusCode)
	}

	var payload errorResponse
	if err := json.Unmarshal(body, &payload); err == nil && payload.Error.Message != "" {
		return fmt.Errorf("openrouter: request failed with status %d: %s", resp.StatusCode, payload.Error.Message)
	}

	text := strings.TrimSpace(string(body))
	if text == "" {
		return fmt.Errorf("openrouter: request failed with status %d", resp.StatusCode)
	}
	return fmt.Errorf("openrouter: request failed with status %d: %s", resp.StatusCode, text)
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Tools    []chatTool    `json:"tools,omitempty"`
	Stream   bool          `json:"stream"`
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
}

type chatTool struct {
	Type     string           `json:"type"`
	Function chatToolFunction `json:"function"`
}

type chatToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type chatToolCall struct {
	ID       string               `json:"id,omitempty"`
	Type     string               `json:"type"`
	Function chatToolCallFunction `json:"function"`
}

type chatToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type streamChunk struct {
	Error   *streamError   `json:"error,omitempty"`
	Choices []streamChoice `json:"choices"`
}

type streamError struct {
	Message string `json:"message"`
}

type streamChoice struct {
	Delta        streamDelta `json:"delta"`
	FinishReason string      `json:"finish_reason"`
}

type streamDelta struct {
	Content   string                `json:"content"`
	ToolCalls []streamToolCallDelta `json:"tool_calls"`
}

type streamToolCallDelta struct {
	Index    int                         `json:"index"`
	ID       string                      `json:"id"`
	Function streamToolCallFunctionDelta `json:"function"`
}

type streamToolCallFunctionDelta struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolCallAccumulator struct {
	id        string
	name      string
	arguments strings.Builder
}

type errorResponse struct {
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}
