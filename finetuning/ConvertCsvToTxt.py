# import pandas as pd

# df = pd.read_csv('processed/scraped.csv')
# content = str(df)
# print(df)
# print(content, file=open('my_file.txt', 'w'))


import  jpype          
jpype.startJVM() 
from asposecells.api import Workbook
from pathlib import Path

# Obtém o diretório atual do script
diretorio_atual = Path(__file__).resolve().parent
print(diretorio_atual)
pasta_processed = diretorio_atual.parent / "processed"

caminho_arquivo = pasta_processed / "scraped.csv"

caminho_arquivo_str = str(caminho_arquivo)

workbook = Workbook(caminho_arquivo_str)

# save as TXT
workbook.save("./processed/SoftcomDataset.txt")