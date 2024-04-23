
# Dependências
# !pip install transformers[torch]
# !pip install tiktoken
# !pip install tokenizers

PATH = './sample_data/'
dados_treino = 'SoftcomDataset.txt'

from tokenizers import ByteLevelBPETokenizer

# Initialize a ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=[PATH+dados_treino], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.encode("Softcom").ids

tokenizer.decode([55, 83, 330, 285])

#neste momento, nosso modelo já possui um tokenizer construído a partir dos dados
#vocab.json, lista dos tokens ordenados por frequência - converte tokens para IDs
#merges.txt - mapeia textos para tokens

# !rm -r ./sample_data/RAW_MODEL
# !mkdir ./sample_data/RAW_MODEL
# tokenizer.save_model(PATH+'RAW_MODEL')

#Usando nosso tokenizer
#https://huggingface.co/docs/transformers/tokenizer_summary

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained(PATH+'RAW_MODEL', max_len=512)

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config)

# quantos parâmetros tem nossa rede neural
model.num_parameters()

#forma simples de se carregar um arquivo bruto como Dataset
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=PATH+dados_treino,
    block_size=128,
)

dataset.examples[:2]

tokenizer.decode(dataset.examples[1]['input_ids'])

'''
Data Collators são estratégias de se construir lotes de dados
para treinar o modelo. Cria listas de amostras a partir do dataset e
permite que o Pytorch aplique a backpropagation adequadamente.
Probability = probabilidade de mascarar tokens da entrada
'''
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.1
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=PATH+'RAW_MODEL',
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model(PATH+'RAW_MODEL')

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model=PATH+'RAW_MODEL',
    tokenizer=PATH+'RAW_MODEL'
)

texto = 'Pergunta <mask>'
fill_mask(texto)