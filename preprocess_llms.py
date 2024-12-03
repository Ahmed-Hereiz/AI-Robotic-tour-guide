import pathlib
from typing import Any
import google.generativeai as genai
from customAgents.ml_models import BaseModels
from customAgents.agent_llm import BaseMultiModal
from mini_rag import query_database

class RAGModel(BaseModels):
    def __init__(self, model_api_key: str, chroma_path: str,):
        super().__init__(model_api_key, chroma_path)

        self.model_api_key = model_api_key
        self.chroma_path = chroma_path

    def inference(self, question: str):
        return query_database(self.model_api_key, self.chroma_path, question)


class AudioMultiModal(BaseMultiModal):
    def __init__(self, api_key: str, model: str, temperature: float = 0.7, safety_settings: Any = None, max_output_tokens: int = None):
        super().__init__(api_key, model, temperature, safety_settings, max_output_tokens)

        genai.configure(api_key=api_key)
        self.audio_model = genai.GenerativeModel(model)

    def multimodal_generate(self, prompt, audio_file_path) -> str:
        
        # audio_file = genai.upload_file(audio_file_path)
        response = self.audio_model.generate_content([
            prompt,
            {
                "mime_type": "audio/mp3",
                 "data": pathlib.Path(audio_file_path).read_bytes()
            },
        ])

        return response.text