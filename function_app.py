import azure.functions as func
from tools.llm_logic import llm_buffer_handler

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="llm_buffer", auth_level=func.AuthLevel.FUNCTION)
def llm_buffer(req: func.HttpRequest) -> func.HttpResponse:
    from tools.llm_logic import llm_buffer_handler
    return llm_buffer_handler(req)

if __name__ == "__main__":
    from tools.load_local_settings import load_local_settings
    load_local_settings()
    from azure.functions import HttpRequest
    import json

    mock_request = HttpRequest(
        method="GET",
        url="/api/llm_buffer",
        headers={},
        params={"query": "look up%20nearby restraunts in%20newberg or"},
        body=None
    )

    result = llm_buffer_handler(mock_request)
    print(result.get_body().decode())
