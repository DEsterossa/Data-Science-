import os
import config

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

MODEL_NAME = "cointegrated/rubert-tiny2"
LLM_MODEL_NAME = "models/gemini-2.5-pro"
RAG_PROMPT_TEMPLATE = """
Используй только следующий контекст, чтобы ответить на вопрос. Если ты не знаешь ответа на основе предоставленного контекста, просто скажи, что не знаешь. Не пытайся выдумать ответ. Отвечай на русском языке.

КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ОТВЕТ:
"""


def create_vector_store_from_file(file_path, vector_store_path):
    """
    Создает и сохраняет векторную базу из ОДНОГО файла.
    """
    print(f"Начало создания VB для файла: {file_path}")

    # Определяем загрузчик в зависимости от расширения файла
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"): # <--- НАШЕ НОВОЕ УСЛОВИЕ
        loader = Docx2txtLoader(file_path)
    else:
        # Можно сделать сообщение более общим
        raise ValueError("Неподдерживаемый формат файла. Пожалуйста, используйте .txt, .pdf или .docx.")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
    vector_db.save_local(vector_store_path)
    print(f"Векторная база сохранена в: {vector_store_path}")
    return True

def get_rag_chain_for_user(vector_store_path):
    """
    Создает RAG-цепочку для конкретной, уже существующей векторной базы пользователя.
    """
    if not os.path.exists(vector_store_path):
        return None # Возвращаем None, если база еще не создана

    embeddings_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.load_local(
        vector_store_path,
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    retriever = vector_db.as_retriever(search_kwargs={'k': 3})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.1)
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain