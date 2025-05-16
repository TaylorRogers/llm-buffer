import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

TRUNCATE_SCRAPED_TEXT = 50000


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


def scrape_page(url, max_chars=TRUNCATE_SCRAPED_TEXT, log=None):
    if log is None:
        log = []
    try:
        log.append(f"🌐 Scraping URL: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
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
        f"You are a helpful assistant tasked with summarizing content relevant to: '{prompt}'."
        f" Provide a concise summary within 500 characters."
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


def build_rag_summary(search_query):
    log = [f"🔁 Starting RAG summary for: {search_query}"]
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        log.append("📦 Building final summary response...")
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": str(structured_data)}
            ],
            temperature=0
        )
        log.append("✅ Final response ready.")
        return final_response.choices[0].message.content
    except Exception as e:
        log.append(f"🔥 Build failed: {e}")
        raise


if __name__ == "__main__":
    from azure.functions import HttpRequest
    from unittest.mock import MagicMock
    import json
    from pathlib import Path

    settings_path = Path(__file__).resolve().parent.parent / "local.settings.json"
    with open(settings_path) as f:
        local_settings = json.load(f)
        for key, value in local_settings.get("Values", {}).items():
            os.environ.setdefault(key, value)

    query = "Latest OpenAI product launches"
    print(build_rag_summary(query))
