import sys
import os
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup

from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage

# Ensure project root is importable
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from openai import OpenAI

TRUNCATE_SCRAPED_TEXT = 50000

@tool("process_content", return_direct=False)
def process_content(url: str) -> str:
    """Processes content from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def scrape_page(url, max_chars=TRUNCATE_SCRAPED_TEXT, log=None):
    if log is None:
        log = []
    try:
        log.append(f"🌐 Scraping URL: {url}")
        headers = {
            "User-Agent": os.getenv("SCRAPER_USER_AGENT", "Mozilla/5.0 (compatible; MyApp/1.0; +http://example.com/bot)")
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        log.append(f"📄 Scraping succeeded: {url}")
        return text[:max_chars * 4]
    except Exception as e:
        log.append(f"❌ Scraping failed for {url}: {e}")
        return f"Error retrieving {url}: {e}"

def summarize_content(content, prompt, client, log=None):
    if log is None:
        log = []
    system_prompt = (
        f"You are a helpful assistant tasked with summarizing content relevant to: '{prompt}'. "
        f"Provide a concise summary within 500 characters."
    )
    try:
        log.append("🧠 Summarizing content...")
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        )
        log.append("✅ Summary generated.")
        return result.choices[0].message.content
    except Exception as e:
        log.append(f"❌ Summarization failed: {e}")
        return "Error in summarization"

def search_google(query, num_results=5, log=None):
    if log is None:
        log = []
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    log.append(f"🔍 Searching Google for: {query}")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num_results
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    log.append("✅ Search completed successfully.")
    return response.json().get("items", [])

def web_search_fn(search_query, log=None):
    """Searches the web or a specific URL and returns a concise, cited summary."""
    if log is None:
        log = []
    log.append(f"🔁 Starting RAG summary for: {search_query}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    url_match = re.search(r"https?://\S+", search_query)
    if url_match:
        url = url_match.group(0)
        if "ix?doc=" in url:
            url = url.split("ix?doc=")[-1]
            if not url.startswith("https://"):
                url = "https://www.sec.gov" + url
            log.append(f"🔁 Rewritten SEC URL: {url}")

        log.append(f"🔗 Direct URL detected: {url}")
        content = scrape_page(url, log=log)
        summary = summarize_content(content, search_query, client, log=log)

        final_prompt = (
            f"Using only the following document content, answer this query: '{search_query}'.\n"
            f"Cite the URL as your source: {url}"
        )
        log.append("📦 Building response from direct document...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": content}
            ],
            temperature=0
        )
        log.append("✅ Final response ready (from URL).")
        return response.choices[0].message.content

    results = search_google(search_query, log=log)
    structured_data = []
    for item in results:
        url = item.get("link")
        snippet = item.get("snippet", "")
        content = scrape_page(url, log=log)
        summary = summarize_content(content, search_query, client, log=log)
        structured_data.append({
            "url": url,
            "snippet": snippet,
            "summary": summary
        })

    final_prompt = (
        f"Using the following search data, answer this query: '{search_query}'.\n"
        f"Cite your sources."
    )
    log.append("📦 Building final summary response (from Google)...")
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": str(structured_data)}
        ],
        temperature=0
    )
    log.append("✅ Final response ready (from search).")
    return final_response.choices[0].message.content

@tool("web_search", return_direct=False)
def web_search(search_query: str) -> str:
    "You are a web searcher. If the query contains a direct file link (e.g., to SEC or HTML files), analyze the file contents directly. Otherwise, search the internet to find the answer."
    return web_search_fn(search_query)

# ───────────────────────────────────────
# Optional Agent for Standalone Execution
# ───────────────────────────────────────
def create_web_search_agent():
    # This system prompt tells the LLM to rewrite shorthand document-URL queries,
    # then immediately invoke the web_search tool with that rewritten prompt.
    system_prompt = (
        "You are a web search agent that does two things in order:\n\n"
        "1. Reformulate shorthand, compact queries containing a document URL into a clear, complete prompt:\n"
        "   - Format exactly: \"Using only the file <URL> <expanded natural-language question>.\"\n"
        "   - Example:\n"
        "       Input:  \"https://sec.gov/...def14a.htm who board directors position\"\n"
        "       Reformulated: \"Using only the file https://sec.gov/...def14a.htm who are the listed board of directors and what is their position.\"\n\n"
        "2. Call the `web_search` tool with that full reformulated prompt as its argument, and return the tool’s response.\n\n"
        "Do not answer the question yourself or modify the tool output."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
    agent = create_openai_tools_agent(llm, [web_search], prompt)

    return AgentExecutor(
        agent=agent,
        tools=[web_search],
        verbose=True
    )

# ───────────────────────────────────────
# CLI for Test or Direct Execution
# ───────────────────────────────────────
if __name__ == "__main__":
    from tools.load_local_settings import load_local_settings
    load_local_settings()
    query = "look up nearby restraunts in newberg or"
    #query = "https://www.sec.gov/Archives/edgar/data/320193/000130817925000008/aapl4359751-def14a.htm who listed board of directors and position"
    #AfterAgentquery = "Using only the file https://www.sec.gov/Archives/edgar/data/320193/000130817925000008/aapl4359751-def14a.htm who are the listed board of directors and what is their position"
    test_log: list[str] = []

    # Option 1: call tool directly
    print("=== TOOL ONLY ===")
    print(web_search_fn(query, log=test_log))
    print("\n=== LOG ===")
    print("\n".join(test_log))

    # Option 2: use the agent
    print("\n=== AGENT EXECUTION ===")
    agent = create_web_search_agent()
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(result["output"])
