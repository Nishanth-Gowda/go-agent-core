package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	prompt := strings.TrimSpace(strings.Join(os.Args[1:], " "))
	if prompt == "" {
		fmt.Fprintln(os.Stderr, "usage: go run ./examples/simple \"Say hi\"")
		os.Exit(2)
	}

	fmt.Printf("go-agent-core scaffold received prompt: %s\n", prompt)
}
