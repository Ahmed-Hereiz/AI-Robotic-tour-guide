from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory
from dotenv import load_dotenv
import os

# api_key for huggingface
load_dotenv()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "api_key"

CHROMA_PATH = "chroma"

# create the llm agent
llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-small", 
                                        task="text2text-generation", 
                                        model_kwargs={"temperature": 0, "max_length": 500}, 
)

# determine DB for retrieval
embedding = HuggingFaceEmbeddings()
vectordb = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding
)
retriever = vectordb.as_retriever()

# create memory
memory = ConversationBufferMemory(
    llm=llm, 
    memory_key="chat_history", 
    output_key='answer', 
    return_messages=True
)

# create a Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(llm, 
                                              retriever=retriever, 
                                              memory=memory
)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Thanks!")
        break
    result = chain.invoke({"question": user_input})

    response = result["answer"]

    print("Chatbot:", response)