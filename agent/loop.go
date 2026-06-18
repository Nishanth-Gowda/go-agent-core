package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"go-agent-core/llm"
	"go-agent-core/tool"
)

const defaultMaxTurns = 16

// LoopOptions configures one low-level agent loop run.
type LoopOptions struct {
	Provider     llm.Provider
	Tools        []tool.Tool
	SystemPrompt string
	MaxTurns     int
	OnEvent      EventHandler
}

// RunLoop appends a user prompt, runs model/tool turns, and returns the updated transcript.
func RunLoop(ctx context.Context, transcript []llm.Message, prompt string, opts LoopOptions) (messages []llm.Message, err error) {
	if opts.Provider == nil {
		return nil, errors.New("agent: provider is required")
	}

	maxTurns := opts.MaxTurns
	if maxTurns == 0 {
		maxTurns = defaultMaxTurns
	}
	if maxTurns < 0 {
		return nil, errors.New("agent: max turns must be non-negative")
	}

	emit := func(event Event) {
		if opts.OnEvent != nil {
			opts.OnEvent(event)
		}
	}

	messages = append([]llm.Message(nil), transcript...)
	messages = append(messages, llm.UserMessage{
		Content: []llm.ContentBlock{llm.TextBlock{Text: prompt}},
	})

	emit(Event{Kind: EventAgentStart})
	defer func() {
		emit(Event{Kind: EventAgentEnd, Err: err})
	}()

	toolsByName := make(map[string]tool.Tool, len(opts.Tools))
	for _, t := range opts.Tools {
		toolsByName[t.Name()] = t
	}

	for turn := 0; turn < maxTurns; turn++ {
		emit(Event{Kind: EventTurnStart, Turn: turn})

		assistant, calls, streamErr := streamAssistant(ctx, opts.Provider, llm.Request{
			SystemPrompt: opts.SystemPrompt,
			Messages:     messages,
			Tools:        tool.Specs(opts.Tools),
		}, turn, emit)
		if streamErr != nil {
			return messages, streamErr
		}

		messages = append(messages, assistant)
		emit(Event{Kind: EventMessageEnd, Turn: turn, Message: assistant})

		if len(calls) == 0 {
			emit(Event{Kind: EventTurnEnd, Turn: turn})
			return messages, nil
		}

		for _, call := range calls {
			result := executeTool(ctx, toolsByName, call, turn, emit)
			messages = append(messages, result)
		}

		emit(Event{Kind: EventTurnEnd, Turn: turn})
	}

	return messages, fmt.Errorf("agent: exceeded max turns (%d)", maxTurns)
}

func streamAssistant(ctx context.Context, provider llm.Provider, req llm.Request, turn int, emit func(Event)) (llm.AssistantMessage, []tool.Call, error) {
	stream, err := provider.Stream(ctx, req)
	if err != nil {
		return llm.AssistantMessage{}, nil, err
	}

	emit(Event{Kind: EventMessageStart, Turn: turn})

	var text strings.Builder
	var content []llm.ContentBlock
	var calls []tool.Call

	flushText := func() {
		if text.Len() == 0 {
			return
		}
		content = append(content, llm.TextBlock{Text: text.String()})
		text.Reset()
	}

	for {
		select {
		case <-ctx.Done():
			return llm.AssistantMessage{}, nil, ctx.Err()
		case event, ok := <-stream:
			if !ok {
				flushText()
				return llm.AssistantMessage{Content: content}, calls, nil
			}

			switch event := event.(type) {
			case llm.TextDeltaEvent:
				if event.Delta == "" {
					continue
				}
				text.WriteString(event.Delta)
				emit(Event{Kind: EventMessageDelta, Turn: turn, Delta: event.Delta})
			case llm.ToolCallEvent:
				flushText()
				block := llm.ToolCallBlock{
					ID:        event.ID,
					Name:      event.Name,
					Arguments: event.Arguments,
				}
				content = append(content, block)
				calls = append(calls, tool.Call{
					ID:        event.ID,
					Name:      event.Name,
					Arguments: event.Arguments,
				})
			case llm.ErrorEvent:
				if event.Err == nil {
					return llm.AssistantMessage{}, nil, errors.New("agent: provider stream error")
				}
				return llm.AssistantMessage{}, nil, event.Err
			default:
				return llm.AssistantMessage{}, nil, fmt.Errorf("agent: unsupported provider event %T", event)
			}
		}
	}
}

func executeTool(ctx context.Context, toolsByName map[string]tool.Tool, call tool.Call, turn int, emit func(Event)) llm.ToolResultMessage {
	emit(Event{Kind: EventToolStart, Turn: turn, ToolCall: call})

	result := llm.ToolResultMessage{
		ToolCallID: call.ID,
		ToolName:   call.Name,
	}

	t, ok := toolsByName[call.Name]
	if !ok {
		result.IsError = true
		result.Content = []llm.ContentBlock{llm.TextBlock{Text: fmt.Sprintf("tool %q not found", call.Name)}}
		emit(Event{Kind: EventToolEnd, Turn: turn, ToolCall: call, ToolResult: result})
		return result
	}

	toolResult, err := t.Execute(ctx, call, func(update tool.Update) {
		emit(Event{Kind: EventToolUpdate, Turn: turn, ToolCall: call, ToolUpdate: update})
	})
	if err != nil {
		result.IsError = true
		result.Content = []llm.ContentBlock{llm.TextBlock{Text: err.Error()}}
		emit(Event{Kind: EventToolEnd, Turn: turn, ToolCall: call, ToolResult: result})
		return result
	}

	result.IsError = toolResult.IsError
	result.Content = toolResult.Content
	if toolResult.CallID != "" {
		result.ToolCallID = toolResult.CallID
	}
	if toolResult.Name != "" {
		result.ToolName = toolResult.Name
	}

	emit(Event{Kind: EventToolEnd, Turn: turn, ToolCall: call, ToolResult: result})
	return result
}
