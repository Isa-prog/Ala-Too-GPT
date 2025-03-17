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
                 "content": "Тебя зовут Ala-Too GPT. Когда нужно, проводи профориентационные тесты, основываясь на потребностях пользователя, но не надоедой с этим. Приоритетнее проводить тест Климова. Проведи тест, задавая вопросы по одному. Затем дай ему итоги теста и на основе результатов советуй им программы обучения в Международном Университете Ала-Тоо. Твои сообщения должны быть предельно точными и короткими."})

# ################################################################################
# ### Step 6
# ################################################################################
#
# def remove_newlines(serie):
#     serie = serie.str.replace('\n', ' ')
#     serie = serie.str.replace('\\n', ' ')
#     serie = serie.str.replace('  ', ' ')
#     serie = serie.str.replace('  ', ' ')
#     return serie
#
# # Create a list to store the text files
# texts = []
#
# # Open the file and read the text
# with open("Ala-Too GPT.txt", "r", encoding="UTF-8") as f:
#     text = f.read()
#
#     # Replace -, _, and #update with spaces.
#     texts.append(("Ala-Too GPT", text))
#
# # Create a dataframe from the list of texts
# df = pd.DataFrame(texts, columns=['fname', 'text'])
#
# # Set the text column to be the raw text with the newlines removed
# df['text'] = df.fname + ". " + remove_newlines(df.text)
# df.to_csv('processed/scraped.csv')
# df.head()
#
# ################################################################################
# ### Step 7
# ################################################################################
#
# # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
# tokenizer = tiktoken.get_encoding("cl100k_base")
#
# df = pd.read_csv('processed/scraped.csv', index_col=0)
# df.columns = ['title', 'text']
#
# # Tokenize the text and save the number of tokens to a new column
# df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
#
# # Visualize the distribution of the number of tokens per row using a histogram
# df.n_tokens.hist()
#
# ################################################################################
# ### Step 8
# ################################################################################
#
# max_tokens = 800
#
# # Function to split the text into chunks of a maximum number of tokens
# def split_into_many(text, max_tokens=max_tokens):
#     # Split the text into sentences
#     sentences = text.split('. ')
#
#     # Get the number of tokens for each sentence
#     n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
#
#     chunks = []
#     tokens_so_far = 0
#     chunk = []
#
#     # Loop through the sentences and tokens joined together in a tuple
#     for sentence, token in zip(sentences, n_tokens):
#
#         # If the number of tokens so far plus the number of tokens in the current sentence is greater
#         # than the max number of tokens, then add the chunk to the list of chunks and reset
#         # the chunk and tokens so far
#         if tokens_so_far + token > max_tokens:
#             chunks.append(". ".join(chunk) + ".")
#             chunk = []
#             tokens_so_far = 0
#
#         # If the number of tokens in the current sentence is greater than the max number of

#         # tokens, go to the next sentence
#         if token > max_tokens:
#             continue
#
#         # Otherwise, add the sentence to the chunk and add the number of tokens to the total
#         chunk.append(sentence)
#         tokens_so_far += token + 1
#
#     # Add the last chunk to the list of chunks
#     if chunk:
#         chunks.append(". ".join(chunk) + ".")
#
#     return chunks
#
# shortened = []
#
# # Loop through the dataframe
# for row in df.iterrows():
#
#     # If the text is None, go to the next row
#     if row[1]['text'] is None:
#         continue
#
#     # If the number of tokens is greater than the max number of tokens, split the text into chunks
#     if row[1]['n_tokens'] > max_tokens:
#         shortened += split_into_many(row[1]['text'])
#
#     # Otherwise, add the text to the list of shortened texts
#     else:
#         shortened.append(row[1]['text'])
#
# ################################################################################
# ### Step 9
# ################################################################################
#
# df = pd.DataFrame(shortened, columns=['text'])
# df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
# df.n_tokens.hist()
#
# ################################################################################
# ### Step 10
# ################################################################################
#
# # Note that you may run into rate limit issues depending on how many files you try to embed
# # Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits
#
# df['embeddings'] = df.text.apply(
#     lambda x: client.embeddings.create(input=[x], model='text-embedding-3-small').data[0].embedding)
# df.to_csv('processed/embeddings.csv')
# df.head()

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
        messages, df, max_len=4000, size="small"  # уменьшено максимальное количество токенов контекста
):
    """
    Create a context for a question by finding the most similar context from the dataframe.

    Combines all user messages into a single string, obtains embeddings for this string,
    and calculates the distances between the string's embeddings and the embeddings in the dataframe.
    Then, it sorts the rows in the dataframe by ascending distance and adds them to the context
    until the total size of the context exceeds max_len.

    Args:
    messages (list): List of messages, including user and assistant messages.
    df (pandas.DataFrame): Dataframe containing texts and their embeddings.
    max_len (int): Maximum length of the context in tokens.
    size (str): Size of the model for embeddings (default is 'small').

    Returns:
    str: The constructed context.
    """

    # Combine all messages into a single string
    combined_messages = " ".join([message['content'] for message in messages if message['role'] == 'user'])

# Get the embeddings for the combined messages
    q_embeddings = client.embeddings.create(input=combined_messages, model='text-embedding-3-small').data[0].embedding

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
        model=GPT_MODEL,
        max_len=4000,  # уменьшено максимальное количество токенов контекста
        size="small",
        debug=False,
        max_tokens=300,  # уменьшено максимальное количество токенов для ответа
        temperature=0.6,  # изменение параметра temperature для генерации более точных ответов
        top_p=1,  # изменение параметра top_p для управления разнообразием ответа
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts.

    This function creates a context for the given question by finding the most similar
    texts in the dataframe and then generates an answer using the OpenAI API. The answer
    is requested to be as concise and precise as possible.

    Args:
    question (str): The question to be answered.
    df (pandas.DataFrame): Dataframe containing texts and their embeddings.
    messages (list): List of messages, including user and assistant messages.
    model (str): The model to be used for generating the answer.
    max_len (int): Maximum length of the context in tokens.
    size (str): Size of the model for embeddings (default is 'small').
    debug (bool): If True, print the context for debugging purposes.
    max_tokens (int): Maximum number of tokens for the answer.
    temperature (float): Sampling temperature for the model's response.
    top_p (float): Nucleus sampling parameter for the model's response.
    stop_sequence (list): Sequence at which to stop generating further tokens.

    Returns:
    str: The generated answer to the question.
    """
    context = create_context(
        messages,
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
            temperature=temperature,
            top_p=top_p,
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
