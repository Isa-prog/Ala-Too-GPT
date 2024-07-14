import os
from ast import literal_eval

import numpy as np
import openai
import pandas as pd
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from openai import OpenAI

client = OpenAI(api_key='')

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

################################################################################
### Step 11
################################################################################

df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)

df.head()


################################################################################
### Step 12
################################################################################

def distances_from_embeddings(embeddings1, embeddings2, distance_metric='cosine'):
    """
    Calculate the distances between two sets of embeddings using cosine similarity.
    """
    if distance_metric == 'cosine':
        embeddings1 = np.array(embeddings1)
        embeddings2 = np.array(embeddings2)

        if embeddings1.shape[0] != embeddings2.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: {embeddings1.shape[0]} != {embeddings2.shape[1]}")

        dot_product = np.dot(embeddings2, embeddings1)
        norm1 = np.linalg.norm(embeddings1)
        norm2 = np.linalg.norm(embeddings2, axis=1)
        return 1 - (dot_product / (norm1 * norm2))


def create_context(
        question, df, max_len=1800, size="small"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = client.embeddings.create(input=[question], model='text-embedding-3-small').data[0].embedding

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values.tolist(),
                                                distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        question,
        df,
        messages,
        model="gpt-4o",
        max_len=1800,
        size="small",
        debug=False,
        max_tokens=150,
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    messages_with_context = messages + [{"role": "system", "content": context}]
    messages_with_context.append({"role": "user", "content": question})

    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages_with_context,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        return ""


################################################################################
### Step 13
################################################################################

while True:
    user_message = input("You: ")
    if user_message.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_message})
    assistant_message = answer_question(user_message, df, messages)
    messages.append({"role": "assistant", "content": assistant_message})
    print(colored(assistant_message, 'green'))
