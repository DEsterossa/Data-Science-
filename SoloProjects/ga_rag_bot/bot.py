import os
import logging
import config  
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from rag_core import create_vector_store_from_file, get_rag_chain_for_user

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

os.makedirs(config.USER_UPLOADS_DIR, exist_ok=True)
os.makedirs(config.USER_VECTOR_STORES_DIR, exist_ok=True)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я ваш персональный ассистент по документам.\n\n"
        "1. Отправьте мне документ (PDF, TXT или DOCX), который вы хотите изучить.\n"
        "2. Я его обработаю и сообщу, когда буду готов.\n"
        "3. После этого вы сможете задавать вопросы по содержанию документа."
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text("Получил ваш документ. Начинаю обработку, это может занять несколько минут...")
    try:
        file_id = update.message.document.file_id
        file = await context.bot.get_file(file_id)
        file_extension = update.message.document.file_name.split('.')[-1].lower()
        
        local_file_path = os.path.join(config.USER_UPLOADS_DIR, f"{chat_id}.{file_extension}")
        await file.download_to_drive(local_file_path)
        
        logging.info(f"Файл от {chat_id} сохранен в {local_file_path}")
        
        vector_store_path = os.path.join(config.USER_VECTOR_STORES_DIR, str(chat_id))
        create_vector_store_from_file(local_file_path, vector_store_path)
        
        await update.message.reply_text("Документ успешно обработан! Теперь вы можете задавать по нему вопросы.")
    except Exception as e:
        logging.error(f"Ошибка при обработке документа от {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка при обработке документа. Пожалуйста, убедитесь, что это .txt, .pdf или .docx файл и попробуйте снова.")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_question = update.message.text
    
    vector_store_path = os.path.join(config.USER_VECTOR_STORES_DIR, str(chat_id))
    
    if not os.path.exists(vector_store_path):
        await update.message.reply_text("Пожалуйста, сначала загрузите документ для анализа.")
        return
        
    await update.message.reply_text("Думаю над вашим вопросом...")
    try:
        rag_chain = get_rag_chain_for_user(vector_store_path)
        if rag_chain is None:
             await update.message.reply_text("Не удалось загрузить вашу базу знаний. Попробуйте перезагрузить документ.")
             return
        answer = rag_chain.invoke(user_question)
        await update.message.reply_text(answer)
    except Exception as e:
        logging.error(f"Ошибка при ответе на вопрос от {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка. Попробуйте еще раз.")

if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_question))
    print("Бот запущен...")
    application.run_polling()