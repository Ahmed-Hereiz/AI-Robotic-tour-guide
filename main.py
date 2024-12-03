from customAgents.agent_llm import BaseLLM
from customAgents.agent_prompt import ReActPrompt
from customAgents.agent_tools import ToolKit, SearchTool, ModelInferenceTool, PythonRuntimeTool
from customAgents.runtime import ReActRuntime
from preprocess_llms import RAGModel, AudioMultiModal
import json


with open("config/llm.json", "r") as f:
    llm_config = json.load(f)

class MainLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, temperature: float, safety_settings=None):
        super().__init__(api_key, model, temperature, safety_settings)

    def llm_generate(self, input: str) -> str:
        return super().generate_response(input,output_style="green")

text = """You are a helpful assistant, you are guide robot to help users. you have tools and you run in a loop don't use the tools use it only if you needed but if you know the answer just answer dircetly"""
multi_modal = AudioMultiModal(api_key=llm_config['api_key'],model=llm_config['model'],temperature=0.7)
audio_respone = multi_modal.multimodal_generate(prompt="transcript what is here ",audio_file_path="demo_commands.mp3")

llm = MainLLM(api_key=llm_config["api_key"], model=llm_config["model"], temperature=0.7)
prompt = ReActPrompt(text=text)
prompt.construct_prompt(query=audio_respone)
# search_tool = SearchTool(tool_name="search tool", description="Tool used to search the internet Used when the model don't know the answer so it can search the internet for the answer")
rag_tool = ModelInferenceTool(tool_name="RAG tool", description="Tool used to query the RAG model Used when the user asks about internal or private data sources, when using you have to refine the input question to the RAG model",model=RAGModel(model_api_key=llm_config["api_key"], chroma_path="chroma"))
python_tool = PythonRuntimeTool(tool_name="python tool", description="Tool used to run python code")
toolkit = ToolKit(tools=[rag_tool, python_tool])

agent = ReActRuntime(llm=llm, prompt=prompt, toolkit=toolkit)
agent.loop(agent_max_steps=10, verbose_tools=True)

