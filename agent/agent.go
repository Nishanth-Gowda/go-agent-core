package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"go-agent-core/llm"
	"go-agent-core/tool"
)

// ErrAgentBusy is returned when a prompt or continuation is started while a run is active.
var ErrAgentBusy = errors.New("agent: already running")

// AgentOptions configures a stateful Agent.
type AgentOptions struct {
	Provider     llm.Provider
	Tools        []tool.Tool
	SystemPrompt string
	MaxTurns     int
	Messages     []llm.Message
}

// AgentState is a snapshot of an Agent's configuration and runtime state.
type AgentState struct {
	Messages         []llm.Message
	Tools            []tool.Tool
	SystemPrompt     string
	MaxTurns         int
	Running          bool
	PendingSteering  int
	PendingFollowUps int
}

type activeRun struct {
	cancel context.CancelFunc
	done   chan struct{}
	err    error
}

type subscription struct {
	id      uint64
	handler EventHandler
}

// Agent is a stateful, concurrency-safe wrapper around the low-level agent loop.
type Agent struct {
	mu sync.Mutex

	provider     llm.Provider
	tools        []tool.Tool
	systemPrompt string
	maxTurns     int
	messages     []llm.Message

	steering []llm.Message
	followUp []llm.Message

	subscriptions []subscription
	nextSubID     uint64

	active  *activeRun
	lastErr error
}

// NewAgent creates a stateful agent with copied top-level message and tool slices.
func NewAgent(opts AgentOptions) *Agent {
	maxTurns := opts.MaxTurns
	if maxTurns == 0 {
		maxTurns = defaultMaxTurns
	}

	return &Agent{
		provider:     opts.Provider,
		tools:        append([]tool.Tool(nil), opts.Tools...),
		systemPrompt: opts.SystemPrompt,
		maxTurns:     maxTurns,
		messages:     append([]llm.Message(nil), opts.Messages...),
	}
}

// State returns a top-level copy of the agent's current state.
func (a *Agent) State() AgentState {
	a.mu.Lock()
	defer a.mu.Unlock()

	return AgentState{
		Messages:         append([]llm.Message(nil), a.messages...),
		Tools:            append([]tool.Tool(nil), a.tools...),
		SystemPrompt:     a.systemPrompt,
		MaxTurns:         a.maxTurns,
		Running:          a.active != nil,
		PendingSteering:  len(a.steering),
		PendingFollowUps: len(a.followUp),
	}
}

// Prompt appends a user prompt and runs until the model stops requesting work.
func (a *Agent) Prompt(ctx context.Context, text string) error {
	message := llm.UserMessage{
		Content: []llm.ContentBlock{llm.TextBlock{Text: text}},
	}
	return a.run(ctx, []llm.Message{message}, false)
}

// Continue resumes from the current transcript without adding a synthetic prompt.
func (a *Agent) Continue(ctx context.Context) error {
	return a.run(ctx, nil, true)
}

func (a *Agent) run(ctx context.Context, initial []llm.Message, continuing bool) error {
	a.mu.Lock()
	if a.active != nil {
		a.mu.Unlock()
		return ErrAgentBusy
	}
	opts := LoopOptions{
		Provider:     a.provider,
		Tools:        append([]tool.Tool(nil), a.tools...),
		SystemPrompt: a.systemPrompt,
		MaxTurns:     a.maxTurns,
		OnEvent:      a.emit,
	}
	if err := validateLoopOptions(opts); err != nil {
		a.mu.Unlock()
		return err
	}

	if continuing {
		var err error
		initial, err = a.continuationMessagesLocked()
		if err != nil {
			a.mu.Unlock()
			return err
		}
	}

	runCtx, cancel := context.WithCancel(ctx)
	run := &activeRun{
		cancel: cancel,
		done:   make(chan struct{}),
	}
	a.active = run
	a.lastErr = nil

	transcript := append([]llm.Message(nil), a.messages...)
	a.mu.Unlock()

	_, err := runLoop(runCtx, transcript, initial, opts, loopHooks{
		onMessage:   a.appendMessage,
		getSteering: a.popSteering,
		getFollowUp: a.popFollowUp,
	})
	cancel()

	a.mu.Lock()
	run.err = err
	a.lastErr = err
	if a.active == run {
		a.active = nil
	}
	close(run.done)
	a.mu.Unlock()

	return err
}

func (a *Agent) continuationMessagesLocked() ([]llm.Message, error) {
	if len(a.messages) == 0 {
		return nil, errors.New("agent: cannot continue without messages")
	}

	last := a.messages[len(a.messages)-1]
	if last == nil {
		return nil, errors.New("agent: cannot continue from a nil message")
	}

	switch last.MessageRole() {
	case llm.RoleUser, llm.RoleTool:
		return nil, nil
	case llm.RoleAssistant:
		if message, ok := popMessage(&a.steering); ok {
			return []llm.Message{message}, nil
		}
		if message, ok := popMessage(&a.followUp); ok {
			return []llm.Message{message}, nil
		}
		return nil, errors.New("agent: cannot continue from an assistant message without queued input")
	default:
		return nil, fmt.Errorf("agent: cannot continue from message role %q", last.MessageRole())
	}
}

// Abort cancels the active provider stream or tool execution, if any.
func (a *Agent) Abort() {
	a.mu.Lock()
	var cancel context.CancelFunc
	if a.active != nil {
		cancel = a.active.cancel
	}
	a.mu.Unlock()

	if cancel != nil {
		cancel()
	}
}

// Wait waits for the active run and returns its error, or the most recent run error when idle.
func (a *Agent) Wait() error {
	a.mu.Lock()
	run := a.active
	if run == nil {
		err := a.lastErr
		a.mu.Unlock()
		return err
	}
	done := run.done
	a.mu.Unlock()

	<-done

	a.mu.Lock()
	err := run.err
	a.mu.Unlock()
	return err
}

// Subscribe registers an event handler and returns an idempotent unsubscribe function.
func (a *Agent) Subscribe(handler EventHandler) func() {
	a.mu.Lock()
	id := a.nextSubID
	a.nextSubID++
	a.subscriptions = append(a.subscriptions, subscription{id: id, handler: handler})
	a.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			a.mu.Lock()
			defer a.mu.Unlock()
			for i, sub := range a.subscriptions {
				if sub.id == id {
					a.subscriptions = append(a.subscriptions[:i], a.subscriptions[i+1:]...)
					return
				}
			}
		})
	}
}

// Steer queues a message for the checkpoint after the current turn.
func (a *Agent) Steer(message llm.Message) {
	a.mu.Lock()
	a.steering = append(a.steering, message)
	a.mu.Unlock()
}

// FollowUp queues a message for when the agent would otherwise stop.
func (a *Agent) FollowUp(message llm.Message) {
	a.mu.Lock()
	a.followUp = append(a.followUp, message)
	a.mu.Unlock()
}

func (a *Agent) appendMessage(message llm.Message) {
	a.mu.Lock()
	a.messages = append(a.messages, message)
	a.mu.Unlock()
}

func (a *Agent) popSteering() (llm.Message, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return popMessage(&a.steering)
}

func (a *Agent) popFollowUp() (llm.Message, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return popMessage(&a.followUp)
}

func popMessage(messages *[]llm.Message) (llm.Message, bool) {
	if len(*messages) == 0 {
		return nil, false
	}
	message := (*messages)[0]
	*messages = (*messages)[1:]
	return message, true
}

func (a *Agent) emit(event Event) {
	a.mu.Lock()
	handlers := make([]EventHandler, 0, len(a.subscriptions))
	for _, sub := range a.subscriptions {
		handlers = append(handlers, sub.handler)
	}
	a.mu.Unlock()

	for _, handler := range handlers {
		handler(event)
	}
}
