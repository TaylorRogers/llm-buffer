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

# ───────────────────────────────────────
# Ensure project root is on sys.path so we can import from tools/
# ───────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.load_local_settings import load_local_settings
from tools.web_search_agent import create_web_search_agent

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def agent_node(state, agent: AgentExecutor, name: str):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def get_members():
    return ["Web_Searcher", "Insight_Researcher"]


def create_supervisor():
    members = get_members()
    system_prompt = (
        f"As a supervisor, oversee the full user query and route it to the right worker:\n"
        f"- If it contains a document URL, use Web_Searcher.\n"
        f"- Otherwise, if deeper analysis is needed, use Insight_Researcher.\n"
        f"Choose one of: {members} or FINISH."
    )
    options = ["FINISH"] + members

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

    llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )


def create_search_agent():
    # trust the imported web_search agent to handle everything
    agent = create_web_search_agent()
    return functools.partial(agent_node, agent=agent, name="Web_Searcher")


def create_insights_researcher_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
    tools = []  # no additional tools needed here
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


def build_graph():
    supervisor_chain = create_supervisor()
    search_node = create_search_agent()
    insights_node = create_insights_researcher_agent()

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("Supervisor", supervisor_chain)
    graph_builder.add_node("Web_Searcher", search_node)
    graph_builder.add_node("Insight_Researcher", insights_node)

    for member in get_members():
        graph_builder.add_edge(member, "Supervisor")

    conditional_map = {m: m for m in get_members()}
    conditional_map["FINISH"] = END
    graph_builder.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)
    graph_builder.set_entry_point("Supervisor")

    return graph_builder.compile()


def run_graph(input_message: str) -> str:
    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage(content=input_message)]})

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
    from load_local_settings import load_local_settings
    load_local_settings()
    query = (
        "Using only the file "
        "https://www.sec.gov/Archives/edgar/data/320193/000130817925000008/aapl4359751-def14a.htm "
        "who are the listed board of directors and what is their position"
    )
    print(run_graph(query))
    """
    test_queries = [
        # 1. SEC document extraction
        "Using only the file https://www.sec.gov/Archives/edgar/data/320193/000130817925000008/aapl4359751-def14a.htm who are the listed board of directors and what is their position",
        # 2. Simple factual web search
        "What is the capital city of Australia?",
        # 3. Multi-topic insight request
        "Analyze recent trends in electric vehicle adoption and summarize the top three factors driving growth",
        # 4. Another document-based query (HTML)
        "From https://en.wikipedia.org/wiki/Python_(programming_language) list the key design philosophies of Python",
        # 5. General opinion synthesis
        "Compare the pros and cons of remote work versus in-office work"
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n=== Test {i} ===")
        print("Query:", q)
        print("Result:")
        print(run_graph(q))
        print("\n" + "-" * 60)

"""
