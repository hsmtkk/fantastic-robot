from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

# https://python.langchain.com/docs/modules/agents/agent_types/openai_tools

llm = ChatOpenAI(streaming=False, verbose=True)

tools = [TavilySearchResults(max_results=1)]

prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_openai_tools_agent(llm, tools, prompt)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

with get_openai_callback() as cb:
    executor.invoke({"input": "Who married with Ootani Shohei?"})
    print(cb)
