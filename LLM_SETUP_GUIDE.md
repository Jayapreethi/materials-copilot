# RAG + LLM Setup Guide

## Auto-Detection (No Configuration Required!)

The system automatically detects and uses available LLM providers in this order:

1. **Ollama** (FREE, local, no setup needed)
2. **Together.ai** (FREE tier, public access)
3. HuggingFace (with API key)
4. OpenAI (with API key)

Just start the dashboard and it will work with whatever is available!

## Option 1: Ollama (Recommended - Completely Free)

**Best for**: Local development, no internet costs, privacy, fastest iteration

### Installation

**Windows/Mac/Linux:**
1. Visit [ollama.ai](https://ollama.ai)
2. Download and install for your OS
3. Open terminal and run:
   ```bash
   ollama serve
   ```
4. In another terminal, pull a model:
   ```bash
   ollama pull llama2
   ```

### Start Dashboard

```bash
cd c:\Projects\rag-co2m
streamlit run dashboard/app.py
```

The dashboard will auto-detect Ollama running at `http://localhost:11434` and use it!

### Available Models

Pull any of these:
```bash
ollama pull llama2           # 7B model, good balance
ollama pull mistral          # 7B, fast
ollama pull neural-chat      # Optimized for chat
ollama pull orca-mini        # Smaller, faster
```

## Option 2: Together.ai (Free Tier - No Setup)

Together.ai offers public API access without authentication initially.

### Start Dashboard

No setup needed! Just run:
```bash
streamlit run dashboard/app.py
```

If Ollama isn't running, it will try Together.ai automatically.

## Option 3: OpenAI (Paid - ~$0.002 per query)

### Get API Key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign in or create account
3. Click "Create new secret key"
4. Copy key (save it!)

### Set Environment Variable

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
streamlit run dashboard/app.py
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=sk-your-key-here
streamlit run dashboard/app.py
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY=sk-your-key-here
streamlit run dashboard/app.py
```

### Pricing

- GPT-3.5-turbo: $0.0015 per 1K input tokens, $0.002 per 1K output tokens
- GPT-4: ~10x more expensive
- Free trial: $5 credit (3-month expiration)

## Option 4: Anthropic Claude (Paid)

### Set API Key

```bash
$env:ANTHROPIC_API_KEY = "sk-ant-your-key"
streamlit run dashboard/app.py
```

### Pricing

- Claude 3 Sonnet: $3 per 1M input tokens, $15 per 1M output tokens

## How to Use AI Generation

1. Open dashboard at [http://localhost:8501](http://localhost:8501)
2. Go to "Semantic Search" tab
3. Check "Generate with AI" checkbox
4. Enter your question
5. Click Search
6. View AI-generated answer with source citations

## Troubleshooting

### "AI not available" shown but Ollama is running
```bash
# Verify Ollama is accessible
curl http://localhost:11434/api/tags

# Restart Ollama in a fresh terminal
ollama serve
```

### Want to see which provider is being used
- Check the badge next to "Generate with AI" checkbox
- Shows "✓ Auto-detected: Ollama" or similar

### Test different providers explicitly
```python
from rag_pipeline import RAGQueryService

# Use Ollama
service = RAGQueryService(llm_provider="ollama")

# Use OpenAI (requires API key)
service = RAGQueryService(llm_provider="openai")

# Use Together
service = RAGQueryService(llm_provider="together")

# Auto-detect (default)
service = RAGQueryService()
```

### Slow responses
- Ollama on CPU can be slower than cloud APIs
- Use smaller model: `ollama pull orca-mini`
- Run on GPU if available

### High costs
- Switch from OpenAI to Ollama (free)
- Use GPT-3.5-turbo instead of GPT-4
- Reduce `max_tokens` in settings

## Dashboard Settings

In the "Semantic Search" tab you can control:

- **Results**: 1-20 documents to retrieve (default 5)
- **Context summary**: Show statistics
- **Concept filter**: Filter by scientific concept
- **Generate with AI**: Toggle LLM response generation

## Python API Usage

```python
from rag_pipeline import RAGQueryService

# Create service (auto-detects provider)
service = RAGQueryService()

# Generate response with AI
response = service.generate_with_llm(
    question="What are the best materials for CO2 capture?",
    top_k=5,
    temperature=0.7,      # Lower = more focused
    max_tokens=500,       # Response length
)

print(response["llm_response"])
print(response["llm_sources"])
```

## Performance Comparison

| Provider | Speed | Cost | Setup | Privacy |
|----------|-------|------|-------|---------|
| Ollama (7B) | Fast | Free | 5 min | 100% |
| Ollama (13B) | Slower | Free | 5 min | 100% |
| OpenAI | Very Fast | $$$ | 1 min | No |
| Claude | Very Fast | $$$ | 1 min | No |
| Together | Fast | Free* | 1 min | No |

*Free tier has rate limits

## Questions?

- Check logs: `streamlit run dashboard/app.py --logger.level=debug`
- Review LLMService docs in `rag_pipeline/llm_service.py`
