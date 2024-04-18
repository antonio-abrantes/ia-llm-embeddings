# Estudos de IA - Como rodar IA localmente e treinar modelos com seus próprios dados


Código implementado para treinar uma IA local usando LLM's e o LM Studio, com dados extraídos da web

### Bibliotecas utilizadas

  ```
  pip install openai matplotlib numpy pandas plotly scipy gradio
    pip install openai[embeddings]
    pip install openai==0.28.1
    pip install openai==0.27.7
    pip install --upgrade openai
  pip install -U scikit-learn
  ```


### Objetivo

Aproveitar do conteúdo disponível no site da empresa para criar um assistente que possa responder questionamentos dentro do contexto apenas dos dados obtidos.

### Como

- Extração de dados textuais de um site através de web scrapping.

- Utilizando os textos do site, utilizei de embeddings para extrair o significado das palavras em um texto através da similaridade semântica.

- O embedding será utilizado como contexto para criar um Conhecimento Customizado ou Custom Knowledge para a LLM escolhida. Assim, o assistente irá responder perguntas com base no conhecimento específico que foi disponibilizado ao assistente.


### Primeiros passos

[Baixar e instalar o LM Studio](https://lmstudio.ai/)
- Baixar um modelo de LLM
- Baixar um modelo de embeddings

### Referência

https://github.com/bixtecnologia/semana-dados-assistente

#### <b>OBS1:</b> Utilizei o projeto mencionado acima, originalmente implementado com o ChatGPT da OpenAI, versão paga, e o adaptei para operar com um modelo local. Foi necessário atualizar e substituir algumas bibliotecas para garantir compatibilidade com a API do LM Studio.
