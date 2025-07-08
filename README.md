# LLM Buffer

**Simple, efficient buffer for LLM-powered downstream M code workflows.**

LLM Buffer is an Azure Functions-based tool designed to act as a lightweight buffer for any system (especially Power BI or M code workflows) that needs to query an LLM with web search but must minimize excess or redundant API calls. Use this as a basic web/LLM buffer layer: your M code can call this function as often as needed, but only fresh requests actually trigger an expensive LLM call. Everything else is pulled from the buffer, enabling fast, cheap, consistent downstream pipelines.

---

## Why?

Modern LLMs, by nature, generate stochastic (non-deterministic) outputs that can reflect subjective information and perspectives drawn from the broader internet or source data. This is powerful for questions where subjectivity, context, or interpretation can be automated where they couldn't before. For example, converting user comments into a product ratings, converting ancient empire names to modern country codes, or determining if a fakebook marketplace item is a good deal. LLM Buffer acts as a buffer in your pipeline to stabilize, cache, and serve these outputs for downstream M code, reducing unnecessary repeat queries and making the subjective outputs more reproducible and useful in analytics workflows.

---

## Features

* **Single HTTP endpoint** with aggressive Azure Blob Storage caching
* **Compatible with Power BI, M code, Python, or any HTTP client**
* **Supports live web search with Google or direct LLM answers**
* **Multi-agent supervisor:** smart agent selection for different queries
* **Zero setup required for downstream clients** (all buffering is handled here)
* Standard Microsoft deployment in the azure cloud

---

## Roadmap / Future Plans

Batch endpoints, table-level categorization, and optimized outputs are **not implemented yet**. These are planned for future updates. The current version focuses on providing a simple, robust buffer layer for single-question calls. See repo issues for status.

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/TaylorRogers/llm-buffer.git
cd llm-buffer
pip install -r requirements.txt
```

### 2. Environment Setup

Export these variables in your shell or define them in `local.settings.json` for Azure Functions:

* `AzureWebJobsStorage` — Azure Storage connection string (for blob cache)
* `OPENAI_API_KEY` — Your OpenAI API key
* `GOOGLE_API_KEY` — Google Custom Search API key
* `GOOGLE_CSE_ID` — Google Custom Search Engine ID

### 3. Run Locally

Requires [Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local):

```bash
func start
```

### 4. Query the API

Example:

```
GET http://localhost:7071/api/llm_buffer?query=What%20is%20the%20capital%20of%20Germany%3F
```

---

## Core API

### Endpoint

`GET /api/llm_buffer`

#### Parameters

* `query` *(string, required)*: Your question.
* `model` *(string, optional)*: LLM model name (default: `gpt-4o-mini`).
* `showlog` *(int, optional)*: Set to `1` to include debug logs.

#### Response

* Returns: `text/plain` answer string (default), or JSON if `showlog=1` or on error.
* Cached answers are returned instantly. New queries trigger the full multi-agent process, with results cached for future use.

#### Error Handling

* Missing/invalid queries return a JSON error message.

---

## How It Works

1. **Request arrives** at `/api/llm_buffer` with a user query.
2. **Cache is checked**—if answer exists, it’s returned immediately.
3. **Supervisor agent** analyzes the question:

   * **Web Search Agent**: For questions needing live/web data (or with URLs), scrapes and summarizes web results using OpenAI.
   * **Insight Researcher**: For general knowledge, uses the LLM only.
   * **Smart Agent**: For complex or ambiguous queries, escalates to a larger LLM model for deep reasoning.
4. **Answer is packaged**, optionally cited, stored in Azure Blob Storage, and returned.

---

## Architecture

```
[HTTP Request]
      |
      v
[llm_buffer_handler] --(Cache hit?)--> [Return Cached Answer]
      |
      v
[Supervisor Agent]
   |        |        |
   v        v        v
[Web]  [Insight]  [Smart]
[Agent][Agent]   [Agent]
   \____   |   ___/
        \  |  /
      [Answer + Cache]
            |
      [HTTP Response]
```

* **Extensible:** Add new agents by updating the supervisor logic.
* **Configurable:** Point to different models, APIs, or cache strategies as needed.

---

## Contributing & Extending

Contributions and suggestions are welcome. This tool is intentionally simple—please keep PRs focused on stability and core buffer functionality.

* Agents/tools live in `tools/`.
* Supervisor logic in `tools/supervisor.py`.
* To add a new agent, create a new tool and register it in the supervisor’s routing logic.
* Submit PRs, open issues, or fork as you like.

---

## Dependencies

* [Azure Functions (Python)](https://pypi.org/project/azure-functions/)
* [OpenAI](https://pypi.org/project/openai/)
* [LangChain](https://pypi.org/project/langchain/)
* [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/)
* [Azure Storage Blob](https://pypi.org/project/azure-storage-blob/)
* [Google Custom Search API](https://developers.google.com/custom-search/v1/overview)
* See `requirements.txt` for the full list.

---

## License

MIT

---

## FAQ / Troubleshooting

* **API keys not working?** Make sure environment variables are set and not checked into git.
* **Cache not updating?** Blob storage connection string must be correct. Try clearing the `llm-cache` container.
* **Want to run outside Azure?** Call the `llm_buffer_handler()` directly from your own script.

---

## Credits

Built by Taylor Rogers.
Inspired by the latest in LLM agent research and real-world workflow automation.
