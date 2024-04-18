# Código das funcionalidades de consulta dos embeddings - Interface v2

import gradio as gr
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

modelEmbedding = "nomic-ai/nomic-embed-text-v1.5-GGUF"
modelLlm = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

# print(df)

# Função para calcular as distâncias entre os embeddings
def distances_from_embeddings(query_embedding, embeddings_list):
    distances = []
    for emb in embeddings_list:
        # Calcula a similaridade cosseno entre o embedding da consulta e cada embedding na lista
        distance = cosine_similarity([query_embedding], [emb])[0][0]
        distances.append(distance)
    return distances

def create_context(question, df, max_len=1800, size="ada"):
    """
    Cria um contexto para uma pergunta encontrando o contexto mais similar no conjunto de embeddings gerado utilizando o Custom Knowledge.
    """

    # Obter a embeddings para a pergunta que foi feita
    q_embeddings = client.embeddings.create(input = [question], model=modelEmbedding).data[0].embedding

    # print(df['embeddings'].values)
    # print(q_embeddings)
    # Obter as distâncias a partir dos embeddings
    distances = distances_from_embeddings(q_embeddings, df['embeddings'].values)
    df['distances'] = distances
    # print(distances)

    returns = []
    cur_len = 0

    # Classifique por distância e adicione o texto ao contexto
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Adicionar o comprimento do texto ao comprimento atual
        cur_len += row['n_tokens'] + 4
        
        # Se o contexto for muito longo, quebre
        if cur_len > max_len:
            break
        
        # Caso contrário, adicione-o ao texto que está sendo retornado
        returns.append(row["text"])

    # Retornar o contexto
    return "\n\n###\n\n".join(returns)

# Funcão que analisa o embedding e cria o contexto conforme a pergunta
def answer_question(
                df=df,
                model=modelLlm,
                question="Onde fica Softcom tecnologia?",
                max_len=1800,
                size="ada",
                debug=False,
                max_tokens=150,
                stop_sequence=None):
      context = create_context(
        question,
        df=df,
        max_len=max_len,
        size=size,
    )
      if debug:
        print("Context:\n" + context)
        print("\n\n")
      try:
        # Criar uma conclusão usando a pergunta e o contexto
        response = client.chat.completions.create(
            model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            messages=[
              {"role": "system", "content": f"Responda as perguntas com base no contexto abaixo, inclusive a lingua, e se a pergunta não puder ser respondida diga \"Eu não sei responder isso\"\n\Contexto: {context}\n\n---\n\nPergunta: {question}\nResposta:"},
              {"role": "user", "content": question}
            ],
            temperature=0.7,
        )
        # print(response)
        return response.choices[0].message
      except Exception as e:
        print(e)

# Testes
# answer = answer_question(question="Qual telefone da Softcom tecnologia?", debug=False)
# answer = answer_question(question="Pode falar algo sobre a cultura da empresa?", debug=False)
# answer = answer_question(df, question="Onde fica a softcom?")

# print(answer.content)

def chatgpt_clone(message, history):
    history= history or []
    s = list(sum(history, ()))
    s.append(message)
    inp = ' '.join(s)
    output=answer_question(question = inp)
    history.append((message, output.content))
    return history, history

def main():
    with gr.Blocks(theme=gr.themes.Soft(),css=".gradio-container {background-color: lightsteelblue}") as block:
        gr.Markdown("""<h1><center> Assistente PaulAI</center></h1>""")
        chatbot=gr.Chatbot(label="Conversa")
        message=gr.Textbox(label="Faça sua pergunta",placeholder="O que você gostaria de saber sobre a Softcom Tecnologia?")
        state = gr.State()
        submit = gr.Button("Perguntar")
        submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])

    block.launch(debug=True)

if __name__ == "__main__":
    main()