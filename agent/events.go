package agent

import (
	"go-agent-core/llm"
	"go-agent-core/tool"
)

// EventKind identifies a lifecycle event emitted by the low-level loop.
type EventKind string

const (
	EventAgentStart   EventKind = "AgentStart"
	EventTurnStart    EventKind = "TurnStart"
	EventMessageStart EventKind = "MessageStart"
	EventMessageDelta EventKind = "MessageDelta"
	EventMessageEnd   EventKind = "MessageEnd"
	EventToolStart    EventKind = "ToolStart"
	EventToolUpdate   EventKind = "ToolUpdate"
	EventToolEnd      EventKind = "ToolEnd"
	EventTurnEnd      EventKind = "TurnEnd"
	EventAgentEnd     EventKind = "AgentEnd"
)

// Event is one lifecycle notification emitted while a loop runs.
type Event struct {
	Kind       EventKind
	Turn       int
	Message    llm.Message
	Delta      string
	ToolCall   tool.Call
	ToolUpdate tool.Update
	ToolResult llm.ToolResultMessage
	Err        error
}

// EventHandler receives lifecycle events.
type EventHandler func(Event)
