package agent

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"go-agent-core/llm"
	"go-agent-core/tool"
)

func TestAgentRejectsConcurrentRuns(t *testing.T) {
	provider := newBlockingProvider()
	agent := NewAgent(AgentOptions{Provider: provider})
	result := make(chan error, 1)
	go func() {
		result <- agent.Prompt(context.Background(), "first")
	}()

	waitForSignal(t, provider.started)
	if err := agent.Prompt(context.Background(), "second"); !errors.Is(err, ErrAgentBusy) {
		t.Fatalf("concurrent Prompt error = %v, want ErrAgentBusy", err)
	}
	if err := agent.Continue(context.Background()); !errors.Is(err, ErrAgentBusy) {
		t.Fatalf("concurrent Continue error = %v, want ErrAgentBusy", err)
	}

	agent.Abort()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("first Prompt error = %v, want context.Canceled", err)
	}
}

func TestAgentAbortCancelsProviderAndWaitReturnsRunError(t *testing.T) {
	provider := newBlockingProvider()
	agent := NewAgent(AgentOptions{Provider: provider})
	result := make(chan error, 1)
	go func() {
		result <- agent.Prompt(context.Background(), "hello")
	}()

	waitForSignal(t, provider.started)
	agent.Abort()

	if err := agent.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Wait error = %v, want context.Canceled", err)
	}
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("Prompt error = %v, want context.Canceled", err)
	}
	if err := agent.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("idle Wait error = %v, want context.Canceled", err)
	}

	state := agent.State()
	if state.Running {
		t.Fatal("State.Running = true after abort")
	}
	if len(state.Messages) != 1 {
		t.Fatalf("message count = %d, want only completed user message", len(state.Messages))
	}
	assertTextMessage(t, state.Messages[0], llm.RoleUser, "hello")

	agent.Abort()
}

func TestAgentAbortCancelsToolExecution(t *testing.T) {
	provider := &fakeProvider{responses: [][]llm.ProviderEvent{{
		llm.ToolCallEvent{ID: "call_1", Name: "blocking"},
	}}}
	blocking := &blockingTool{started: make(chan struct{})}
	agent := NewAgent(AgentOptions{
		Provider: provider,
		Tools:    []tool.Tool{blocking},
	})
	result := make(chan error, 1)
	go func() {
		result <- agent.Prompt(context.Background(), "run tool")
	}()

	waitForSignal(t, blocking.started)
	agent.Abort()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("Prompt error = %v, want context.Canceled", err)
	}

	state := agent.State()
	if len(state.Messages) != 3 {
		t.Fatalf("message count = %d, want user, assistant, and canceled tool result", len(state.Messages))
	}
	assertToolResult(t, state.Messages[2], "call_1", "blocking", true, context.Canceled.Error())
}

func TestAgentSteersOneMessagePerTurn(t *testing.T) {
	provider := &fakeProvider{responses: [][]llm.ProviderEvent{
		{llm.TextDeltaEvent{Delta: "first"}},
		{llm.TextDeltaEvent{Delta: "second"}},
		{llm.TextDeltaEvent{Delta: "third"}},
	}}
	agent := NewAgent(AgentOptions{Provider: provider})
	agent.Steer(userMessage("steer one"))
	agent.Steer(userMessage("steer two"))

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("Prompt returned error: %v", err)
	}

	if len(provider.requests) != 3 {
		t.Fatalf("provider calls = %d, want 3", len(provider.requests))
	}
	assertRequestTexts(t, provider.requests[0], []string{"start"})
	assertRequestTexts(t, provider.requests[1], []string{"start", "first", "steer one"})
	assertRequestTexts(t, provider.requests[2], []string{"start", "first", "steer one", "second", "steer two"})

	state := agent.State()
	if state.PendingSteering != 0 || state.PendingFollowUps != 0 {
		t.Fatalf("pending queues = steering %d, follow-ups %d", state.PendingSteering, state.PendingFollowUps)
	}
}

func TestAgentFollowUpRunsAfterNaturalStop(t *testing.T) {
	provider := &fakeProvider{responses: [][]llm.ProviderEvent{
		{llm.TextDeltaEvent{Delta: "first"}},
		{llm.TextDeltaEvent{Delta: "second"}},
	}}
	agent := NewAgent(AgentOptions{Provider: provider})
	var once sync.Once
	agent.Subscribe(func(event Event) {
		if event.Kind == EventTurnEnd {
			once.Do(func() {
				agent.FollowUp(userMessage("follow up"))
			})
		}
	})

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("Prompt returned error: %v", err)
	}
	if len(provider.requests) != 2 {
		t.Fatalf("provider calls = %d, want 2", len(provider.requests))
	}
	assertRequestTexts(t, provider.requests[1], []string{"start", "first", "follow up"})
}

func TestAgentSteeringTakesPriorityOverFollowUp(t *testing.T) {
	provider := &fakeProvider{responses: [][]llm.ProviderEvent{
		{llm.TextDeltaEvent{Delta: "first"}},
		{llm.TextDeltaEvent{Delta: "second"}},
		{llm.TextDeltaEvent{Delta: "third"}},
	}}
	agent := NewAgent(AgentOptions{Provider: provider})
	agent.FollowUp(userMessage("follow up"))
	agent.Steer(userMessage("steer"))

	if err := agent.Prompt(context.Background(), "start"); err != nil {
		t.Fatalf("Prompt returned error: %v", err)
	}
	assertRequestTexts(t, provider.requests[1], []string{"start", "first", "steer"})
	assertRequestTexts(t, provider.requests[2], []string{"start", "first", "steer", "second", "follow up"})
}

func TestAgentContinue(t *testing.T) {
	t.Run("from user message", func(t *testing.T) {
		provider := &fakeProvider{responses: [][]llm.ProviderEvent{{llm.TextDeltaEvent{Delta: "continued"}}}}
		agent := NewAgent(AgentOptions{
			Provider: provider,
			Messages: []llm.Message{userMessage("existing")},
		})

		if err := agent.Continue(context.Background()); err != nil {
			t.Fatalf("Continue returned error: %v", err)
		}
		assertRequestTexts(t, provider.requests[0], []string{"existing"})
	})

	t.Run("assistant with queued input", func(t *testing.T) {
		provider := &fakeProvider{responses: [][]llm.ProviderEvent{{llm.TextDeltaEvent{Delta: "continued"}}}}
		agent := NewAgent(AgentOptions{
			Provider: provider,
			Messages: []llm.Message{assistantMessage("existing")},
		})
		agent.Steer(userMessage("new direction"))

		if err := agent.Continue(context.Background()); err != nil {
			t.Fatalf("Continue returned error: %v", err)
		}
		assertRequestTexts(t, provider.requests[0], []string{"existing", "new direction"})
	})

	t.Run("invalid transcript", func(t *testing.T) {
		agent := NewAgent(AgentOptions{Provider: &fakeProvider{}})
		if err := agent.Continue(context.Background()); err == nil {
			t.Fatal("Continue returned nil error for empty transcript")
		}

		agent = NewAgent(AgentOptions{
			Provider: &fakeProvider{},
			Messages: []llm.Message{assistantMessage("done")},
		})
		if err := agent.Continue(context.Background()); err == nil {
			t.Fatal("Continue returned nil error after assistant without queued input")
		}
	})
}

func TestAgentSubscribersAndStateSnapshots(t *testing.T) {
	provider := &fakeProvider{responses: [][]llm.ProviderEvent{
		{llm.TextDeltaEvent{Delta: "first"}},
		{llm.TextDeltaEvent{Delta: "second"}},
	}}
	agent := NewAgent(AgentOptions{
		Provider: provider,
		Tools:    []tool.Tool{fakeTool{name: "fake"}},
	})
	var calls []string
	unsubscribe := agent.Subscribe(func(event Event) {
		if event.Kind == EventMessageEnd {
			calls = append(calls, "first")
			if got := len(agent.State().Messages); got == 0 {
				t.Error("State had no committed messages during MessageEnd")
			}
		}
	})
	agent.Subscribe(func(event Event) {
		if event.Kind == EventMessageEnd {
			calls = append(calls, "second")
		}
	})

	if err := agent.Prompt(context.Background(), "one"); err != nil {
		t.Fatalf("first Prompt returned error: %v", err)
	}
	if !reflect.DeepEqual(calls, []string{"first", "second"}) {
		t.Fatalf("subscriber calls = %#v", calls)
	}

	unsubscribe()
	unsubscribe()
	calls = nil
	if err := agent.Prompt(context.Background(), "two"); err != nil {
		t.Fatalf("second Prompt returned error: %v", err)
	}
	if !reflect.DeepEqual(calls, []string{"second"}) {
		t.Fatalf("subscriber calls after unsubscribe = %#v", calls)
	}

	state := agent.State()
	state.Messages[0] = userMessage("changed")
	state.Tools[0] = nil
	got := agent.State()
	assertTextMessage(t, got.Messages[0], llm.RoleUser, "one")
	if got.Tools[0] == nil {
		t.Fatal("mutating State.Tools changed agent state")
	}
}

type blockingProvider struct {
	started chan struct{}
}

func newBlockingProvider() *blockingProvider {
	return &blockingProvider{started: make(chan struct{})}
}

func (p *blockingProvider) Stream(ctx context.Context, req llm.Request) (<-chan llm.ProviderEvent, error) {
	close(p.started)
	stream := make(chan llm.ProviderEvent)
	go func() {
		<-ctx.Done()
		close(stream)
	}()
	return stream, nil
}

type blockingTool struct {
	started chan struct{}
}

func (t *blockingTool) Name() string           { return "blocking" }
func (t *blockingTool) Description() string    { return "blocks until canceled" }
func (t *blockingTool) Schema() map[string]any { return nil }
func (t *blockingTool) Execute(ctx context.Context, call tool.Call, update tool.UpdateFunc) (tool.Result, error) {
	close(t.started)
	<-ctx.Done()
	return tool.Result{}, ctx.Err()
}

func userMessage(text string) llm.UserMessage {
	return llm.UserMessage{Content: []llm.ContentBlock{llm.TextBlock{Text: text}}}
}

func assistantMessage(text string) llm.AssistantMessage {
	return llm.AssistantMessage{Content: []llm.ContentBlock{llm.TextBlock{Text: text}}}
}

func assertRequestTexts(t *testing.T, request llm.Request, want []string) {
	t.Helper()
	got := make([]string, 0, len(request.Messages))
	for _, message := range request.Messages {
		content := message.MessageContent()
		if len(content) != 1 {
			t.Fatalf("message content count = %d, want 1", len(content))
		}
		block, ok := content[0].(llm.TextBlock)
		if !ok {
			t.Fatalf("content block type = %T, want llm.TextBlock", content[0])
		}
		got = append(got, block.Text)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("request texts = %#v, want %#v", got, want)
	}
}

func waitForSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for signal")
	}
}
