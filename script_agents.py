from customAgents.agent_llm import SimpleStreamLLM
from customAgents.agent_prompt import SimplePrompt
from customAgents.runtime import SimpleRuntime


def AI_agent_sys_offline(api_key, model, temperature, query, file_paths):
    with open("define-robot-tasks.txt", "r") as f:
        prompt_text = f.read()

    llm = SimpleStreamLLM(api_key=api_key, model=model, temperature=temperature)

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            text = file.read()

            prompt = SimplePrompt(text=text + prompt_text)
            prompt.construct_prompt(query=query)  

            agent = SimpleRuntime(llm=llm, prompt=prompt)

            agent.loop()


api_key = "AIzaSyBaJ4tFib2D0qpTvpjLIVf4bkA5k0YwXY0"
model = "gemini-1.5-flash"
temperature = 0.7

query = "Erzählen Sie mir Informationen über die Pyramiden"

file_paths = [
    "The-Rosetta-Stone.txt",
    "The-Mask-of-Tutankhamun.txt",
    "The-Karnak-temple.txt",
    "The-Great-Pyramids-of-Giza.txt",
    "The-Valley-of-the-kings.txt"
]

AI_agent_sys_offline(api_key, model, temperature, query, file_paths)
