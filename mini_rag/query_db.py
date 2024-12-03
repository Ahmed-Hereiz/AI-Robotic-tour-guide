import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def query_database(api_key: str, chroma_path: str, question: str):
    
    genai.configure(api_key=api_key)
    
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    vectordb = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_model
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7,
        convert_system_message_to_human=True
    )

    docs = vectordb.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Based on the following context, please answer the question.
    Context: {context}
    
    Question: {question}
    """
    
    response = llm.invoke(prompt)
    answer = f"You: {question}\nChatbot: {response.content}"
    return answer
