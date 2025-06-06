# LLM Buffer

This Azure Functions app exposes a single `/llm_buffer` endpoint that answers questions using OpenAI models and optional web search. Results are cached in Azure Blob Storage to avoid repeated API calls.

## Environment Variables
- `AzureWebJobsStorage` – connection string for the Azure storage account used for caching
- `OPENAI_API_KEY` – API key for OpenAI models
- `GOOGLE_API_KEY` / `GOOGLE_CSE_ID` – keys for Google Custom Search (used by the web search agent)

## Running Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Create `local.settings.json` with the environment variables above.
3. Use `func start` to run the Azure Function locally.

A simple request can be sent with:
```
GET /api/llm_buffer?query=What+is+the+capital+of+Germany&model=gpt-4o-mini
```

Cached responses are stored under the `llm-cache` container in blob storage, keyed by the query and model name.

