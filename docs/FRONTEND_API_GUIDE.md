# Frontend API Usage Guide

## Recommended Endpoints for Frontend Development

### 1. **Primary Endpoint: `/invoke`** 🎯

**Use this for**: Standard agent interactions, production applications

```typescript
// Frontend TypeScript example
interface UniversalAgentInput {
  payload: Record<string, any>;
  streaming?: boolean;
  prompt_version?: string;
  temperature?: number;
  max_tokens?: number;
  metadata?: Record<string, any>;
}

interface UniversalAgentResponse {
  output: any;
  agent: string;
  prompt_version: string;
  streaming: boolean;
  metadata?: Record<string, any>;
  validation_info?: Record<string, any>;
}

// Alpha Agent Example
const alphaResponse = await fetch('/api/v1/agents/alpha/invoke', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    payload: {
      query: "What is machine learning?",
      context: "Educational context for beginners"
    },
    streaming: false,
    temperature: 0.7
  })
});

// Beta Agent Example
const betaResponse = await fetch('/api/v1/agents/beta/invoke', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    payload: {
      problem: "How to optimize database performance?",
      domain: "Software Engineering",
      requirements: ["Low latency", "High throughput"],
      constraints: "Limited memory budget"
    },
    streaming: false
  })
});
```

### 2. **Streaming Endpoint: `/chat`** ⚡

**Use this for**: Real-time chat interfaces, streaming responses

```typescript
interface AgentRequest {
  input: string;
  context?: string;
  streaming?: boolean;
  prompt_version?: string;
  temperature?: number;
  max_tokens?: number;
  metadata?: Record<string, any>;
}

// Non-streaming chat
const chatResponse = await fetch('/api/v1/agents/alpha/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    input: "Explain quantum computing",
    context: "For a technical audience",
    streaming: false
  })
});

// Streaming chat
const streamingResponse = await fetch('/api/v1/agents/alpha/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    input: "Tell me about AI trends",
    streaming: true
  })
});

// Handle streaming response
const reader = streamingResponse.body?.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader!.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data === '[DONE]') return;
      if (data.startsWith('[ERROR]')) {
        console.error('Stream error:', data);
        return;
      }
      // Handle streaming data
      console.log('Received:', data);
    }
  }
}
```

## Schema Discovery

Get agent input schemas dynamically:

```typescript
// Get schema for any agent
const schemaResponse = await fetch('/api/v1/agents/alpha/schema');
const schema = await schemaResponse.json();

console.log('Alpha agent schema:', schema.data);
// Use this to build dynamic forms or validate inputs
```

## Frontend Framework Examples

### React Hook Example

```tsx
import { useState, useCallback } from 'react';

interface AgentHookOptions {
  agent: string;
  streaming?: boolean;
}

export function useAgent({ agent, streaming = false }: AgentHookOptions) {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const invoke = useCallback(async (payload: Record<string, any>) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/v1/agents/${agent}/invoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          payload,
          streaming
        })
      });

      if (!response.ok) {
        throw new Error(`Agent request failed: ${response.status}`);
      }

      const data = await response.json();
      setResponse(data);
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [agent, streaming]);

  return { invoke, loading, response, error };
}

// Usage in component
function ChatComponent() {
  const { invoke, loading, response, error } = useAgent({ agent: 'alpha' });

  const handleSubmit = async (query: string) => {
    await invoke({ query, context: 'chat interface' });
  };

  return (
    <div>
      {loading && <div>Processing...</div>}
      {error && <div>Error: {error}</div>}
      {response && <div>Response: {response.data.output}</div>}
    </div>
  );
}
```

### Vue.js Composable Example

```typescript
import { ref, reactive } from 'vue';

export function useAgent(agentName: string) {
  const loading = ref(false);
  const response = ref(null);
  const error = ref<string | null>(null);

  const invoke = async (payload: Record<string, any>) => {
    loading.value = true;
    error.value = null;

    try {
      const res = await fetch(`/api/v1/agents/${agentName}/invoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ payload })
      });

      const data = await res.json();
      response.value = data;
      return data;
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Unknown error';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  return {
    invoke,
    loading: readonly(loading),
    response: readonly(response),
    error: readonly(error)
  };
}
```

## Error Handling

The API returns standardized error responses:

```typescript
interface APIError {
  detail: string | {
    error: string;
    agent: string;
    validation_errors: any[];
    expected_schema: string;
  };
}

// Handle validation errors
try {
  const response = await fetch('/api/v1/agents/beta/invoke', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      payload: { invalid: 'data' }
    })
  });

  if (!response.ok) {
    const error: APIError = await response.json();

    if (response.status === 422 && typeof error.detail === 'object') {
      // Handle validation errors
      console.log('Validation failed for agent:', error.detail.agent);
      console.log('Expected schema:', error.detail.expected_schema);
      console.log('Validation errors:', error.detail.validation_errors);
    }
  }
} catch (err) {
  console.error('Request failed:', err);
}
```

## Summary

- **Use `/invoke` for 95% of your frontend needs** - it's the most flexible and future-proof
- **Use `/chat` only if you need streaming** for real-time chat interfaces
- **Both endpoints support all agents** automatically through dynamic discovery
- **Schema validation is built-in** and provides helpful error messages
- **Responses are standardized** across all agents for consistent frontend handling

This simplified API structure makes frontend development much easier while maintaining all the power and flexibility of the multi-agent system.
