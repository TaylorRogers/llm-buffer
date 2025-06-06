import azure.functions as func
from tools.llm_logic import llm_buffer_handler

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="llm_buffer", auth_level=func.AuthLevel.FUNCTION)
def llm_buffer(req: func.HttpRequest) -> func.HttpResponse:
    from tools.llm_logic import llm_buffer_handler
    return llm_buffer_handler(req)

@app.route(route="http_trigger", auth_level=func.AuthLevel.FUNCTION)
def deploy1(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("ok", status_code=200)
