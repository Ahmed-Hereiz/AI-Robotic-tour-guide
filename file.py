from customAgents.agent_llm import SimpleStreamLLM
from customAgents.agent_prompt import ReActPrompt
from customAgents.agent_tools import ToolKit, SearchTool
from customAgents.runtime import ReActRuntime
import json

with open("config/llm.json", "r") as f:
    llm_config = json.load(f)

text = """You are a helpful assistant, you are guide robot to help users."""

llm = SimpleStreamLLM(api_key=llm_config["api_key"], model=llm_config["model"], temperature=0.7)
prompt = ReActPrompt(text=text)
prompt.construct_prompt(query="What is the weather in Tokyo?")
print(prompt)
search_tool = SearchTool(tool_name="search tool", description="Tool used to search the internet")
toolkit = ToolKit(tools=[search_tool])
agent = ReActRuntime(llm=llm, prompt=prompt, toolkit=toolkit)

