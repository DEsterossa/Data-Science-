EMBEDDER_MODEL_NAME = "cointegrated/rubert-tiny2"
LLM_MODEL_NAME = "models/gemini-2.5-flash"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

USER_UPLOADS_DIR = "user_data/uploads"
USER_VECTOR_STORES_DIR = "user_data/vector_stores"

RAG_PROMPT_TEMPLATE = """
Используй только следующий контекст, чтобы ответить на вопрос. Если ты не знаешь ответа на основе предоставленного контекста, просто скажи, что не знаешь. Не пытайся выдумать ответ. Отвечай на русском языке.

КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ОТВЕТ:
"""