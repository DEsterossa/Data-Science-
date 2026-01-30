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
from pathlib import Path
from datetime import datetime

MODEL_NAME = "cointegrated/rubert-tiny2"
LLM_MODEL_NAME = "models/gemini-2.5-flash"
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
    Создает или дополняет векторную базу пользователя содержимым одного файла.
    Если для пользователя уже существует FAISS-индекс, новые чанки добавляются к нему.
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

    # Формируем идентификатор документа и имя файла для метаданных
    path_obj = Path(file_path)
    file_name = path_obj.name
    # doc_id: базовое имя файла + timestamp, чтобы отличать загрузки с одинаковым именем
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    doc_id = f"{path_obj.stem}_{timestamp}"

    # Добавляем метаданные документа ко всем исходным Document
    for doc in documents:
        # не перезаписываем уже существующие ключи, если они есть
        doc.metadata = doc.metadata or {}
        doc.metadata.setdefault("doc_id", doc_id)
        doc.metadata.setdefault("file_name", file_name)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Гарантируем наличие метаданных на уровне чанков
    for chunk in chunks:
        chunk.metadata = chunk.metadata or {}
        chunk.metadata.setdefault("doc_id", doc_id)
        chunk.metadata.setdefault("file_name", file_name)

    embeddings_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Если индекс уже существует, загружаем и дополняем его новыми документами
    if os.path.exists(vector_store_path) and os.path.isdir(vector_store_path):
        index_faiss = os.path.join(vector_store_path, "index.faiss")
        index_pkl = os.path.join(vector_store_path, "index.pkl")
        if os.path.exists(index_faiss) and os.path.exists(index_pkl):
            print(f"Обнаружен существующий индекс для пути: {vector_store_path}, выполняю дозапись.")
            try:
                vector_db = FAISS.load_local(
                    vector_store_path,
                    embeddings_model,
                    allow_dangerous_deserialization=True,
                )
                vector_db.add_documents(chunks)
            except Exception as e:
                # Если по какой-то причине не удалось загрузить старый индекс, логируем и создаем новый
                print(f"Не удалось загрузить существующий индекс ({e}). Переcоздаю индекс с нуля.")
                vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
        else:
            # Папка есть, но индекса нет — создаем новый
            print(f"Папка {vector_store_path} существует, но файлов индекса нет. Создаю новый индекс.")
            vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
    else:
        # Индекс для пользователя отсутствует — создаем новый
        print(f"Индекс для пути {vector_store_path} не найден. Создаю новый индекс.")
        os.makedirs(vector_store_path, exist_ok=True)
        vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings_model)

    vector_db.save_local(vector_store_path)
    print(f"Векторная база сохранена/обновлена в: {vector_store_path}")
    return True

def get_rag_chain_for_user(vector_store_path, doc_id: str | None = None):
    """
    Создает RAG-цепочку для конкретной, уже существующей векторной базы пользователя.
    В дальнейшем doc_id можно использовать для фильтрации по документу.
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

    # Пока используем простой retriever без фильтрации.
    # В будущем здесь можно учитывать doc_id (через фильтры или постфильтрацию по metadata["doc_id"]).
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