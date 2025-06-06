import functools
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_smart_agent():
    """Return an agent that uses a powerful model for deep reasoning."""
    system_prompt = (
        "You are a Smart Agent that performs complex reasoning tasks "
        "carefully and concisely."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
    agent = create_openai_tools_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=True)
