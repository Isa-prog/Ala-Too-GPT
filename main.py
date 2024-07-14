import os
import numpy as np
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from openai import OpenAI

# Убедись, что ты заменил 'your-api-key-here' на свой реальный ключ API
client = OpenAI(api_key='')

# Задать идентификатор векторного стора
vector_store_id = "vs_d0PUrlfrEboLHkaecnglOp0R"

# Функция для получения эмбеддингов из векторного стора
def get_embeddings_from_vector_store(vector_store_id, text):
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small',
        vector_store=vector_store_id
    )
    return response['data'][0]['embedding']

# Функция для создания контекста
def create_context(question, vector_store_id, max_len=1800):
    """
    Создание контекста для вопроса, находя наиболее похожий контекст из векторного стора
    """
    # Получение встраиваний для вопроса из векторного стора
    q_embeddings = get_embeddings_from_vector_store(vector_store_id, question)

    # Получение встраиваний всех документов из векторного стора
    response = client.vector_stores.retrieve(
        vector_store=vector_store_id,
        limit=1000  # Указать подходящее значение лимита
    )
    documents = response['data']

    # Преобразование документов в DataFrame
    df = pd.DataFrame(documents)

    # Проверка полученных данных
    print(f"\nRetrieved {len(documents)} documents from vector store.")

    # Вывод первых нескольких документов для проверки
    print("Sample documents:")
    for i, doc in enumerate(documents[:5]):
        print(f"Document {i + 1}:")
        print(f"Text: {doc['text'][:200]}...")  # Вывод первых 200 символов текста
        print(f"Embedding: {doc['embedding'][:5]}...")  # Вывод первых 5 значений эмбеддинга
        print()

    # Вычисление косинусного сходства (через скалярное произведение)
    df['similarity'] = df['embedding'].apply(lambda x: np.dot(q_embeddings, x))

    returns = []
    cur_len = 0

    # Сортировка по сходству и добавление текста в контекст, пока контекст не станет слишком длинным
    for i, row in df.sort_values('similarity', ascending=False).iterrows():
        text_len = len(row['text'].split())
        cur_len += text_len + 4

        if cur_len > max_len:
            break

        returns.append(row["text"])

    # Возвращение контекста
    return "\n\n###\n\n".join(returns)

def answer_question(question, model="text-davinci-003", max_len=1800, max_tokens=150, stop_sequence=None):
    """
    Ответ на вопрос на основе наиболее похожего контекста из векторного стора
    """
    context = create_context(question, vector_store_id, max_len=max_len)

    # Выводим контекст на печать для проверки
    print("\nGenerated Context:\n", context)

    try:
        # Создание завершения с использованием вопроса и контекста
        response = client.completions.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(e)
        return ""

GPT_MODEL = "gpt-4o"

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

messages = []
messages.append({"role": "system",
                 "content": "Тебя зовут Ala-Too GPT. Когда нужно, проводи профоиентационные тесты основываясь на потребностях пользователя. Приоритетнее проводить тест Климова. Проведи тест задавая вопросы по одному. Затем дай ему итоги теста и на основе результатов советуй им программы обучение в Международном Университете Ала-Тоо. Твои сообщения должны быть предельно точными и короткими."})

while True:
    user_message = input("You: ")
    if user_message.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_message})
    assistant_message = answer_question(user_message)
    messages.append({"role": "assistant", "content": assistant_message})
    print(colored(assistant_message, 'green'))
