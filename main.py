from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from customAgents.agent_llm import SimpleStreamLLM, BaseLLM
from customAgents.agent_prompt import SimplePrompt, ReActPrompt
from customAgents.agent_tools import ModelInferenceTool, PythonRuntimeTool, SearchTool, ToolKit
from customAgents.runtime import SimpleRuntime, ReActRuntime
from prompt import script_generator_prompt
from preprocess_llms import RAGModel, AudioMultiModal
import json
import os
import uvicorn  
from gtts import gTTS
import uuid
import base64

app = FastAPI()

with open("config/llm.json", "r") as f:
    llm_config = json.load(f)

class MainLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, temperature: float, safety_settings=None):
        super().__init__(api_key, model, temperature, safety_settings)

    def llm_generate(self, input: str) -> str:
        return super().generate_response(input,output_style="green")

llm = SimpleStreamLLM(api_key=llm_config['api_key'], model=llm_config['model'], temperature=0.3)

class UserInput(BaseModel):
    description: str

class AudioPath(BaseModel):
    path: str

@app.post("/refine-scripts/")
async def refine_scripts(user_input: UserInput):
    output_dict = {}
    description = user_input.description
    for filename in os.listdir('refine_text/'):
        prompt = SimplePrompt(text=script_generator_prompt)
        offline_agent = SimpleRuntime(llm=llm, prompt=prompt)
        if filename.endswith('.txt'):
            with open(os.path.join('refine_text/', filename), 'r') as file:
                script = file.read()
                prompt.construct_prompt(placeholder_dict={"{script}": script, "{description}": description})
                new_script = offline_agent.loop()
                output_dict[filename] = new_script


    os.makedirs('output_aud', exist_ok=True)
    for filename, script in output_dict.items():
        audio_filename = os.path.splitext(filename)[0] + '.mp3'
        audio_path = os.path.join('output_aud', audio_filename)
        
        tts = gTTS(text=script, lang='en')
        tts.save(audio_path)

    print("Audio should be sent through the api")

    return output_dict

@app.post("/get_audio_files/")
async def get_audio_files():
    if not os.path.exists('output_aud'):
        raise HTTPException(status_code=404, detail="No audio files directory found")
    
    audio_files = [f for f in os.listdir('output_aud') if f.endswith('.mp3')]
    
    if not audio_files:
        raise HTTPException(status_code=404, detail="No audio files found")

    response = {}
    
    for audio_file in audio_files:
        file_path = os.path.join('output_aud', audio_file)
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            response[audio_file] = audio_b64

    print("audio files should be sent...")

    return response


@app.post("/audio_interface/")
async def audio_interface(audio_file: UploadFile = File(...)):
    
    text = """You are a helpful assistant, you are guide robot to help users. you have tools and you run in a loop don't use the tools use it only if you needed but if you know the answer just answer dircetly, NOTE Always give answer to the user either using the RAG or your base knowledge don't ever say you don't you have to always give answer for the user in the language he talked to you with"""
    multi_modal = AudioMultiModal(api_key=llm_config['api_key'], model=llm_config['model'], temperature=0.7)
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    # Save uploaded file to temp directory with unique name
    temp_input_audio = os.path.join("temp", f"input_{uuid.uuid4()}.mp3")
    audio_bytes = await audio_file.read()
    with open(temp_input_audio, "wb") as f:
        f.write(audio_bytes)
    
    audio_response = multi_modal.multimodal_generate(prompt="transcript what is here ", audio_file_path=temp_input_audio)
    os.remove(temp_input_audio)

    llm = MainLLM(api_key=llm_config["api_key"], model=llm_config["model"], temperature=0.7)
    prompt = ReActPrompt(text=text)
    prompt.construct_prompt(query=audio_response)
    
    rag_tool = ModelInferenceTool(tool_name="RAG tool", description="Tool used to query the RAG model Used when the user asks about internal or private data sources, when using you have to refine the input question to the RAG model", model=RAGModel(model_api_key=llm_config["api_key"], chroma_path="chroma"))
    python_tool = PythonRuntimeTool(tool_name="python tool", description="Tool used to run python code")
    toolkit = ToolKit(tools=[rag_tool, python_tool])

    agent = ReActRuntime(llm=llm, prompt=prompt, toolkit=toolkit)
    output = agent.loop(agent_max_steps=10, verbose_tools=True)
    
    # Generate response audio with unique name
    temp_audio_file = os.path.join("temp", f"response_{uuid.uuid4()}.mp3")
    tts = gTTS(output, lang='en')
    tts.save(temp_audio_file)
    
    with open(temp_audio_file, "rb") as audio:
        audio_bytes = audio.read()
    
    os.remove(temp_audio_file)
    
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    return {"audio_response": audio_b64}

@app.post("/text_interface/")
async def text_interface(user_input: UserInput):
    text = """You are a helpful assistant, you are guide robot to help users. you have tools and you run in a loop don't use the tools use it only if you needed but if you know the answer just answer dircetly, NOTE Always give answer to the user either using the RAG or your base knowledge don't ever say you don't you have to always give answer for the user in the language he talked to you with"""

    llm = MainLLM(api_key=llm_config["api_key"], model=llm_config["model"], temperature=0.7)
    prompt = ReActPrompt(text=text)
    prompt.construct_prompt(query=user_input.description)
    
    rag_tool = ModelInferenceTool(tool_name="RAG tool", description="Tool used to query the RAG model Used when the user asks about internal or private data sources, when using you have to refine the input question to the RAG model", model=RAGModel(model_api_key=llm_config["api_key"], chroma_path="chroma"))
    python_tool = PythonRuntimeTool(tool_name="python tool", description="Tool used to run python code")
    toolkit = ToolKit(tools=[rag_tool, python_tool])

    agent = ReActRuntime(llm=llm, prompt=prompt, toolkit=toolkit)
    output = agent.loop(agent_max_steps=10, verbose_tools=True)
    return {"response": output}

@app.post("/clear_memory/")
async def clear_memory():
    return {"status": "memory cleared"}


if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8000)
