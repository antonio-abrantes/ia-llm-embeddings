from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

completion = client.chat.completions.create(
  model="local-model", # this field is currently unused
  messages=[
    {"role": "system", "content": "Responder apenas em portugues br."},
    {"role": "user", "content": "Circular Import"}
  ],
  temperature=0.7,
)

# print(completion.choices[0].message)
print(completion.choices[0].message.content)

# test = "Circular Import: O erro menciona a possibilidade de uma importação circular. Isso ocorre quando um módulo importa outro que, por sua vez, tenta importar o primeiro. Verifique se há importações circulares em seus arquivos Python e tente reorganizá-las para evitar essa situação"

# def get_embedding(text, model="local-model"):
#    text = text.replace("\n", " ")
#    return client.embeddings.create(input = [text], model=model).data[0].embedding

# print(get_embedding(test))