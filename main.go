package main

// Agent loop pseudocode:
//
//      messages = [{ role: "user", content: prompt }]
//
//      loop:
//          response = call_api(messages)
//          append response message to messages
//
//          if response has no tool_calls:
//              print response.content
//              exit
//
//          for each tool_call in response.tool_calls:
//              result = execute_tool(tool_call)
//              append {
//                  role: "tool",
//                  tool_call_id: tool_call.id,
//                  content: result
//              } to messages

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func main() {
	loadDotEnv(".env")

	var prompt string
	flag.StringVar(&prompt, "p", "", "Prompt to send to the model")
	flag.Parse()

	interactive := prompt == "" // no -p flag: interactive REPL session

	// Local-first defaults: works out of the box against Ollama.
	// Override via environment to target any OpenAI-compatible backend, e.g.:
	//   BASE_URL=https://openrouter.ai/api/v1 API_KEY=sk-or-... MODEL=anthropic/claude-haiku-4.5
	baseURL := os.Getenv("BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1"
	}

	apiKey := os.Getenv("API_KEY")
	if apiKey == "" {
		apiKey = "ollama" // Ollama ignores the key; real backends need a real one
	}

	model := os.Getenv("MODEL")
	if model == "" {
		model = "qwen3:1.7b"
	}

	fmt.Fprintf(os.Stderr, "agent: model=%s base_url=%s\n", model, baseURL)

	client := openai.NewClient(option.WithAPIKey(apiKey), option.WithBaseURL(baseURL))

	tools := []openai.ChatCompletionToolUnionParam{
		openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "Read",
			Description: openai.String("Read and return the contents of a file"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"file_path": map[string]any{
						"type":        "string",
						"description": "The path to the file to read",
					},
				},
				"required": []string{"file_path"},
			},
		}),
		openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "Write",
			Description: openai.String("Write content to a file"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"file_path": map[string]any{
						"type":        "string",
						"description": "The path of the file to write to",
					},
					"content": map[string]any{
						"type":        "string",
						"description": "The content to write to the file",
					},
				},
				"required": []string{"file_path", "content"},
			},
		}),
		openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "Bash",
			Description: openai.String("Execute a shell command"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "The command to execute",
					},
				},
				"required": []string{"command"},
			},
		}),
	}

	// messages = [system prompt] + first user turn (one-shot) or nothing (REPL)
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(systemPrompt),
	}
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

	// readInput prompts on stderr and returns the next non-empty stdin line.
	// ok is false on EOF (Ctrl-D).
	readInput := func() (string, bool) {
		for {
			fmt.Fprint(os.Stderr, "\n> ")
			if !scanner.Scan() {
				return "", false
			}
			line := strings.TrimSpace(scanner.Text())
			if line != "" {
				return line, true
			}
		}
	}

	if interactive {
		fmt.Fprintln(os.Stderr, "interactive mode — type a request, /exit or Ctrl-D to quit")
		// The first user message must exist BEFORE the first model call,
		// otherwise we'd send a conversation with no user turn in it.
		input, ok := readInput()
		if !ok || input == "/exit" || input == "/quit" {
			return
		}
		messages = append(messages, openai.UserMessage(input))
	} else {
		messages = append(messages, openai.UserMessage(prompt))
	}

	for {
		// run the agent loop until the model replies without tool calls
		messages = converse(client, model, tools, messages)

		if !interactive {
			break
		}

		input, ok := readInput()
		if !ok || input == "/exit" || input == "/quit" {
			break
		}
		messages = append(messages, openai.UserMessage(input))
	}
}

// systemPrompt primes the model to act as an agent instead of asking questions.
const systemPrompt = `You are a coding agent working in the user's current directory.
You have tools: Read (read a file), Write (create or overwrite a file), Bash (run a shell command).
Use your tools to inspect files and make changes yourself instead of asking the user for details.
If the user's request mentions a file, read it with the Read tool before answering.
When the task is done, reply concisely.`

// converse runs the agent loop until the model replies without tool calls,
// printing that final reply to stdout. It returns the updated message history.
func converse(client openai.Client, model string, tools []openai.ChatCompletionToolUnionParam,
	messages []openai.ChatCompletionMessageParamUnion,
) []openai.ChatCompletionMessageParamUnion {
	for {
		// response = call_api(messages)
		resp, err := client.Chat.Completions.New(context.Background(),
			openai.ChatCompletionNewParams{
				Model:    model,
				Messages: messages,
				Tools:    tools,
			})
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		if len(resp.Choices) == 0 {
			panic("No choices in response")
		}

		assistantMsg := resp.Choices[0].Message

		// append response message to messages
		messages = append(messages, assistantMsg.ToParam())

		// if response has no tool_calls:
		//     print response.content
		if len(assistantMsg.ToolCalls) == 0 {
			fmt.Print(assistantMsg.Content)
			return messages
		}

		// for each tool_call in response.tool_calls:
		for _, toolCall := range assistantMsg.ToolCalls {
			// result = execute_tool(tool_call)
			result := executeTool(toolCall)

			// append { role: "tool", tool_call_id: tool_call.id, content: result } to messages
			messages = append(messages, openai.ToolMessage(result, toolCall.ID))
		}
	}
}

// execute_tool(tool_call)
func executeTool(toolCall openai.ChatCompletionMessageToolCallUnion) string {
	switch toolCall.Function.Name {
	case "Read":
		var args struct {
			FilePath string `json:"file_path"`
		}
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
			return fmt.Sprintf("error unmarshaling arguments: %v", err)
		}

		content, err := os.ReadFile(args.FilePath)
		if err != nil {
			return fmt.Sprintf("error reading file: %v", err)
		}
		return string(content)

	case "Write":
		var args struct {
			FilePath string `json:"file_path"`
			Content  string `json:"content"`
		}
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
			return fmt.Sprintf("error unmarshaling arguments: %v", err)
		}

		if err := os.WriteFile(args.FilePath, []byte(args.Content), 0o644); err != nil {
			return fmt.Sprintf("error writing file: %v", err)
		}
		return fmt.Sprintf("File %s written successfully", args.FilePath)

	case "Bash":
		var args struct {
			Command string `json:"command"`
		}
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
			return fmt.Sprintf("error unmarshaling arguments: %v", err)
		}
		return executeBash(args.Command)

	default:
		return fmt.Sprintf("unknown tool: %s", toolCall.Function.Name)
	}
}

// maxOutputLen caps how much command output is sent back to the model so a
// chatty command can't blow up the context window.
const maxOutputLen = 10000

// executeBash runs command via bash, merges stdout+stderr, and enforces a
// timeout so a hung command can't stall the agent loop forever.
func executeBash(command string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "bash", "-c", command)
	out, err := cmd.CombinedOutput()

	result := string(out)
	if len(result) > maxOutputLen {
		result = result[:maxOutputLen] + "\n... (output truncated)"
	}
	if err != nil {
		return fmt.Sprintf("Command failed: %v\nOutput:\n%s", err, result)
	}
	if result == "" {
		return "(no output)"
	}
	return result
}

func loadDotEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for i, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			fmt.Fprintf(os.Stderr, "warning: %s:%d: ignoring malformed line %q\n",
				path, i+1, line)
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if len(value) >= 2 && (value[0] == '"' || value[0] == '\'') && value[len(value)-1] == value[0] {
			value = value[1 : len(value)-1]
		}
		if _, exists := os.LookupEnv(key); !exists {
			os.Setenv(key, value)
		}
	}
}
