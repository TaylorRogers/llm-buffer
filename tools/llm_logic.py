import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import hashlib
import json
import os
from datetime import datetime

import azure.functions as func
from azure.storage.blob import BlobServiceClient
from tools.supervisor import ask


def get_params(req, log):
    query = req.params.get("query")
    model = req.params.get("model") or "gpt-4o-mini"
    showlog = req.params.get("showlog") or "0"
    if not query:
        log.append("❌ Missing 'query' parameter.")
        raise ValueError("Missing query")
    log.append(f"🤖 Query: {query}")
    log.append(f"🧠 Model: {model}")
    return query.strip(), model.strip(), showlog.strip()


def create_blob_name(query, model):
    base_key = f"{model}::{query}".lower().strip()
    hash_key = hashlib.sha256(base_key.encode()).hexdigest()
    return f"{model}/{hash_key}.json", hash_key


def get_connection_string(log):
    conn_str = os.getenv("AzureWebJobsStorage")
    if not conn_str or "AccountKey=" not in conn_str:
        log.append("❌ AzureWebJobsStorage is missing or malformed.")
        raise ValueError("Missing or invalid connection string")
    log.append("🔑 Connection string retrieved.")
    return conn_str


def blob_exists(conn_str, blob_name, log):
    try:
        client = BlobServiceClient.from_connection_string(conn_str)
        blob = client.get_blob_client(container="llm-cache", blob=blob_name)
        if blob.exists():
            log.append("📦 Cache hit — returning previous response.")
            data = blob.download_blob().readall()
            return json.loads(data)
        return None
    except Exception as e:
        log.append(f"⚠️ Blob existence check failed: {e}")
        return None


def upload_blob(conn_str, blob_name, data, tags, log):
    try:
        client = BlobServiceClient.from_connection_string(conn_str)
        blob = client.get_blob_client(container="llm-cache", blob=blob_name)
        blob.upload_blob(json.dumps(data), overwrite=True)
        log.append("📤 Stored response in cache.")
    except Exception as e:
        log.append(f"❌ Blob upload failed: {e}")
        raise


def build_response(result, code=200):
    return func.HttpResponse(json.dumps(result, indent=2), mimetype="application/json", status_code=code)


def llm_buffer_handler(req: func.HttpRequest) -> func.HttpResponse:
    log = []
    log.append("🔁 Starting LLM query buffer process...")
    try:
        query, model, showlog = get_params(req, log)
        conn_str = get_connection_string(log)
        blob_name, hash_key = create_blob_name(query, model)

        cached = blob_exists(conn_str, blob_name, log)
        if cached:
            # Normalize if cached is list or contains a list in "answer"
            if isinstance(cached, list):
                cached = {"answer": "\n".join(str(x) for x in cached)}
            elif isinstance(cached, dict) and isinstance(cached.get("answer"), list):
                cached["answer"] = "\n".join(str(x) for x in cached["answer"])

            if showlog == "1":
                return build_response({"result": cached, "log": log})
            return func.HttpResponse(cached["answer"], mimetype="text/plain", status_code=200)

        answer_text = ask(query)

        record = {
            "query": query,
            "model": model,
            "answer": answer_text,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        tags = {"model": model, "query_hash": hash_key}
        upload_blob(conn_str, blob_name, record, tags, log)

        if showlog == "1":
            return build_response({"result": record, "log": log})
        return func.HttpResponse(answer_text, mimetype="text/plain", status_code=200)

    except Exception as e:
        log.append(f"🔥 Aborted due to error: {e}")
        return build_response({"error": str(e)})


if __name__ == "__main__":
    from tools.load_local_settings import load_local_settings
    load_local_settings()

    from azure.functions import HttpRequest
    from unittest.mock import MagicMock

    # Example 1
    mock_req = MagicMock(spec=HttpRequest)
    mock_req.params = {"query": "What is the capital of Germany", "model": "gpt-4o-mini"}
    response = llm_buffer_handler(mock_req)
    print("=== ANSWER ===\n", response.get_body().decode())

    # Example 2
    mock_req2 = MagicMock(spec=HttpRequest)
    prompt2 = (
        "Using only the file https://www.sec.gov/Archives/edgar/data/320193/000130817925000008/"
        "aapl4359751-def14a.htm, who are the listed board of directors and their positions? llama222"
    )
    mock_req2.params = {"query": prompt2, "model": "gpt-4o-mini", "showlog": "0"}
    response2 = llm_buffer_handler(mock_req2)
    print("\n=== ANSWER 2 ===\n", response2.get_body().decode())

