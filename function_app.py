import hashlib
import json
import os
from datetime import datetime
from urllib.parse import quote_plus

import azure.functions as func
from azure.storage.blob import BlobServiceClient
from openai import OpenAI

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# ---------- Subfunctions ----------

def get_params(req, log):
    query = req.params.get("query")
    model = req.params.get("model", "gpt-4o-mini")
    if not query:
        log.append("❌ Missing 'query' parameter.")
        raise ValueError("Missing query")
    log.append(f"🤖 Query: {query}")
    log.append(f"🧠 Model: {model}")
    return query.strip(), model.strip()

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

def call_openai(query, model, log):
    try:
        log.append("📡 Calling OpenAI API...")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
        )
        answer = response.choices[0].message.content
        log.append("✅ Received OpenAI response.")
        return {"answer": answer}
    except Exception as e:
        log.append(f"❌ OpenAI call failed: {e}")
        raise

def upload_blob(conn_str, blob_name, data, tags, log):
    try:
        client = BlobServiceClient.from_connection_string(conn_str)
        blob = client.get_blob_client(container="llm-cache", blob=blob_name)
        blob.upload_blob(json.dumps(data), overwrite=True)
        # Blob tags are not supported in your current account type (non-GPv2), so this is skipped
        # blob.set_blob_tags(tags)
        log.append("📤 Stored response in cache.")
    except Exception as e:
        log.append(f"❌ Blob upload failed: {e}")
        raise

def build_response(result, log, code=200, include_log=True):
    if include_log:
        body = {
            "result": result,
            "log": log
        }
    else:
        if isinstance(result, dict) and "answer" in result:
            body = result["answer"]
        else:
            body = result
    return func.HttpResponse(json.dumps(body, indent=2), mimetype="application/json", status_code=code)

# ---------- Main Entry Point ----------

@app.route(route="llm_buffer", auth_level=func.AuthLevel.FUNCTION)
def llm_buffer(req: func.HttpRequest) -> func.HttpResponse:
    log = ["🔁 Starting LLM query buffer process..."]
    include_log = req.params.get("showlog", "false").lower() == "true"
    try:
        query, model = get_params(req, log)
        conn_str = get_connection_string(log)
        blob_name, hash_key = create_blob_name(query, model)

        cached = blob_exists(conn_str, blob_name, log)
        if cached:
            return build_response(cached, log, include_log=include_log)

        result = call_openai(query, model, log)
        record = {
            "query": query,
            "model": model,
            "answer": result["answer"],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        tags = {
            "model": model,
            "query_hash": hash_key
        }

        upload_blob(conn_str, blob_name, record, tags, log)
        return build_response(record, log, include_log=include_log)

    except Exception as e:
        log.append(f"🔥 Aborted due to error: {e}")
        return build_response({"error": str(e)}, log, code=500, include_log=include_log)
