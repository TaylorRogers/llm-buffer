import functools, operator, sys
from typing import Annotated, Sequence, TypedDict
from pathlib import Path

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import StateGraph, END

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.load_local_settings import load_local_settings
from tools.web_search_agent import create_web_search_agent
from tools.smart_agent import create_smart_agent

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    steps: int  # track supervisor recursions
    web_search_count: int  # limit web search usage

def agent_node(state, agent: AgentExecutor, name: str):
    result = agent.invoke(state)
    update = {"messages": [HumanMessage(content=result["output"], name=name)]}
    if name == "Web_Searcher":
        update["web_search_count"] = state.get("web_search_count", 0) + 1
    return update

def get_members():
    return ["Web_Searcher", "Insight_Researcher", "Smart_Agent"]

def create_supervisor():
    members = get_members()
    options = ["FINISH"] + members
    system_prompt = (
        f"As a supervisor, oversee the full user query and route it to the right worker:\n"
        f"- If it contains a document URL, use Web_Searcher.\n"
        f"- Otherwise, if basic reasoning suffices, use Insight_Researcher.\n"
        f"- For very challenging queries, you MAY use Smart_Agent, but only if strictly necessary as it is expensive.\n"
        f"- Web_Searcher may be used only once per query.\n"
        f"Choose one of: {members} or FINISH."
    )

    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {"title": "Next", "anyOf": [{"enum": options}]}
            },
            "required": ["next"],
        },
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Who should act next? Select one of: {options}"),
    ]).partial(options=str(options), members=", ".join(members))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

def supervisor_node(state, supervisor_chain):
    steps = state.get("steps", 0) + 1
    if steps > 2:
        return {"next": "FINISH", "messages": state["messages"], "steps": steps}
    out = supervisor_chain.invoke(state)
    out["steps"] = steps
    if state.get("web_search_count", 0) >= 1 and out.get("next") == "Web_Searcher":
        out["next"] = "Insight_Researcher"
    return out

def create_search_agent():
    agent = create_web_search_agent()
    return functools.partial(agent_node, agent=agent, name="Web_Searcher")

def create_insights_researcher_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)
    tools = []
    system_prompt = (
        "You are an Insight Researcher. Do step-by-step analysis:\n"
        "1. Identify key topics in the full user query.\n"
        "2. Search or reason as needed.\n"
        "3. Summarize insights with sources."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return functools.partial(agent_node, agent=AgentExecutor(agent=agent, tools=tools, verbose=True), name="Insight_Researcher")

def create_smart_agent_node():
    agent = create_smart_agent()
    return functools.partial(agent_node, agent=agent, name="Smart_Agent")

def build_graph():
    supervisor_chain = create_supervisor()
    search_node = create_search_agent()
    insights_node = create_insights_researcher_agent()
    smart_node = create_smart_agent_node()

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("Supervisor", functools.partial(supervisor_node, supervisor_chain=supervisor_chain))
    graph_builder.add_node("Web_Searcher", search_node)
    graph_builder.add_node("Insight_Researcher", insights_node)
    graph_builder.add_node("Smart_Agent", smart_node)

    for member in get_members():
        graph_builder.add_edge(member, "Supervisor")

    conditional_map = {m: m for m in get_members()}
    conditional_map["FINISH"] = END
    graph_builder.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)
    graph_builder.set_entry_point("Supervisor")

    return graph_builder.compile()

def run_graph(input_message: str) -> str:
    graph = build_graph()
    result = graph.invoke({
        "messages": [HumanMessage(content=input_message)],
        "steps": 0,
        "web_search_count": 0,
    })

    content = result["messages"][1].content
    output, refs = "", []
    for line in content.splitlines():
        if line.strip().startswith("[^"):
            refs.append(line.strip())
        else:
            output += line + "\n"
    if refs:
        output += "\n**References:**\n" + "\n".join(refs)
    return output

def ask(input_message: str) -> str:
    return run_graph(input_message)

if __name__ == "__main__":
    load_local_settings()
    query = (
        "Using only the file "
        "https://www.sec.gov/Archives/edgar/data/320193/000130817925000008/aapl4359751-def14a.htm "
        "who are the listed board of directors and what is their position"
    )
    print(run_graph(query))
