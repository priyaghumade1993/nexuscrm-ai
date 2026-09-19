# Lab 08: Alternative LLM Providers

> **Note:** This is a conceptual comparison only. NexusCRM AI uses **OpenAI** as its sole LLM provider.  
> Google Gemini, Anthropic Claude, and Ollama are **not implemented** in this project by design.  
> The abstraction layer (`BaseLLMProvider`) in `backend/app/agents/llm_provider.py` makes adding them straightforward.

---

## Why NexusCRM Uses OpenAI

The project specification required a single LLM provider to keep scope focused. OpenAI was chosen because:

1. **LangChain integration** — `ChatOpenAI` is LangChain's best-supported provider; function calling, tool use, and structured output work out of the box.
2. **GPT-4o quality** — state-of-the-art reasoning for the intent classification and tool selection tasks.
3. **GPT-4o-mini cost** — sub-tasks (guardrail semantic check, output validation) use `gpt-4o-mini` at ~1/10th the cost.
4. **OpenAI embeddings** — `text-embedding-3-small` pairs naturally with the rest of the stack.

---

## Provider Comparison Matrix

| Property | OpenAI GPT-4o | Google Gemini 1.5 Pro | Anthropic Claude 3.5 Sonnet | Ollama (local) |
|---|---|---|---|---|
| **Context window** | 128K tokens | 1M tokens | 200K tokens | Varies by model |
| **Function calling** | ✓ Excellent | ✓ Good | ✓ Excellent | ✓ (Llama 3.1+) |
| **Structured output** | ✓ JSON mode | ✓ JSON mode | ✓ tool_choice | Varies |
| **LangChain support** | `ChatOpenAI` | `ChatGoogleGenerativeAI` | `ChatAnthropic` | `ChatOllama` |
| **Cost (input/1M tokens)** | $2.50 (4o) | $1.25 (1.5 Pro) | $3.00 (Sonnet) | Free (local) |
| **Cost (output/1M tokens)** | $10.00 (4o) | $5.00 (1.5 Pro) | $15.00 (Sonnet) | Free (local) |
| **Data privacy** | Sent to OpenAI | Sent to Google | Sent to Anthropic | Fully local |
| **Self-hosted option** | ✗ (Azure OpenAI) | ✗ (Vertex AI) | ✗ (AWS Bedrock) | ✓ (your hardware) |
| **Rate limits** | High (Tier 3+) | High | Moderate | None |
| **Embeddings** | `text-embedding-3-small` | `embedding-001` | Not offered | `nomic-embed-text` |

---

## How NexusCRM Would Add Each Provider

### Google Gemini

```python
# Install: pip install langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agents.llm_provider import BaseLLMProvider

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True,  # Gemini quirk
        )
    
    def get_llm(self):
        return self._llm
    
    def get_model_name(self) -> str:
        return self._llm.model
```

**Key difference:** Gemini requires `convert_system_message_to_human=True` because the Gemini API doesn't have a native system message role — LangChain converts it to a human turn automatically. The 1M context window is valuable for large document RAG.

**Gotcha:** Gemini's function calling format differs from OpenAI's. LangChain normalizes this, but streaming + tools requires LangChain ≥ 0.2.0.

---

### Anthropic Claude

```python
# Install: pip install langchain-anthropic
from langchain_anthropic import ChatAnthropic
from app.agents.llm_provider import BaseLLMProvider

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self._llm = ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=0,
            max_tokens=4096,  # Required for Anthropic — no default
        )
    
    def get_llm(self):
        return self._llm
    
    def get_model_name(self) -> str:
        return self._llm.model
```

**Key difference:** Anthropic requires explicit `max_tokens`. Claude 3.5 Sonnet has excellent instruction-following for structured tool use — comparable to GPT-4o. The 200K context makes it well-suited for whole-codebase reasoning.

**Gotcha:** Claude's tool calling format uses `tool_choice` differently from OpenAI's `function_call`. LangChain normalizes this but test thoroughly when migrating.

---

### Ollama (Local / Self-Hosted)

```python
# Install: pip install langchain-ollama
# Requires: Ollama running locally — ollama serve
from langchain_ollama import ChatOllama
from app.agents.llm_provider import BaseLLMProvider

class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
        )
    
    def get_llm(self):
        return self._llm
    
    def get_model_name(self) -> str:
        return self._llm.model
```

**Key difference:** Requires Ollama to be installed and running locally. No API key. Hardware requirements:
- `llama3.1:8b` — 8GB VRAM (or ~12GB RAM for CPU)
- `llama3.1:70b` — 40GB+ VRAM
- `mistral:7b` — 8GB VRAM; faster than Llama 3.1 8B on some tasks

**Quality tradeoff:** Local 8B models are significantly worse than GPT-4o/Claude 3.5 for:
- Complex multi-step reasoning
- Following precise tool call schemas
- Instruction adherence in multi-agent coordination

For NexusCRM's intent classification + tool orchestration, a local 7-8B model would reduce accuracy by ~15-20% on the evaluation dataset (estimated).

**When Ollama makes sense:**
- Air-gapped environments (no internet access)
- Extremely sensitive data (healthcare, defense)
- Very high volume with budget constraints (>10M tokens/day)
- Development/testing (no API costs)

---

## Provider Factory (How It Would Work)

The `get_default_provider()` function in `backend/app/agents/llm_provider.py` already reads `LLM_PROVIDER` from settings. Adding a new provider is just a `match` case:

```python
def get_default_provider() -> BaseLLMProvider:
    provider = settings.LLM_PROVIDER.lower()  # "openai" | "gemini" | "anthropic" | "ollama"
    
    match provider:
        case "openai":
            return OpenAIProvider(api_key=settings.OPENAI_API_KEY, ...)
        case "gemini":
            return GeminiProvider(api_key=settings.GOOGLE_API_KEY)
        case "anthropic":
            return AnthropicProvider(api_key=settings.ANTHROPIC_API_KEY)
        case "ollama":
            return OllamaProvider(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)
        case _:
            raise ValueError(f"Unknown LLM provider: {provider}")
```

The rest of the codebase (LangGraph nodes, tools, RAG) is provider-agnostic — they receive a `BaseChatModel` from LangChain and call `.invoke()` / `.ainvoke()`. No other code changes.

---

## Evaluation Results by Provider (Conceptual)

These are estimated relative scores based on published benchmarks, not actual NexusCRM evaluations:

| Metric | GPT-4o | GPT-4o-mini | Claude 3.5 Sonnet | Gemini 1.5 Pro | Llama 3.1 8B |
|---|---|---|---|---|---|
| Intent accuracy | ~95% | ~88% | ~94% | ~91% | ~78% |
| Tool call precision | ~96% | ~85% | ~95% | ~89% | ~72% |
| Safety rate | ~99% | ~97% | ~99% | ~97% | ~90% |
| Tokens/$ | 1× | ~10× | ~0.8× | ~2× | ∞ (free) |
| Latency (p50) | ~800ms | ~400ms | ~900ms | ~600ms | ~2000ms (CPU) |

**Conclusion:** GPT-4o is the best balance of quality and cost for NexusCRM's workload. `gpt-4o-mini` is used for guardrail checks where lower cost matters more than peak accuracy.

---

## Switching Providers in NexusCRM

To switch from OpenAI to any provider above (once implemented):

```bash
# In .env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key-here

# Or for local Ollama
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

No code changes required. The evaluation suite (`backend/app/evaluation/runner.py`) can be run against any provider to measure real-world quality differences.
