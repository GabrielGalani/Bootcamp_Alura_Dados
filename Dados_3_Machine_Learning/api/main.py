
import pickle as pk
import numpy as np
import pandas as pd
from fastapi import FastAPI

#definindo funções
SEED = 80
np.random.seed(SEED)

def criar_idade(df):
    faixa_etaria = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    faixa_etaria_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

    df['faixa_idade'] = pd.cut(x=df['pessoa_idade'], bins=faixa_etaria, labels=faixa_etaria_labels)
    df['faixa_idade'] = df['faixa_idade'].astype('int64')

    return df

def criar_valor(df):
    
    faixa_valor = [499, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 35000, 36000]
    faixa_valor_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70]

    df['faixa_emprestimo'] = pd.cut(x=df['vl_total'], bins=faixa_valor, labels=faixa_valor_labels)
    df['faixa_emprestimo'] = df['faixa_emprestimo'].astype('int64')
    return df

#definindo API
app = FastAPI()

with open('API/one_hot_encoder.pkl', 'rb') as f:
    one_hot_enc = pk.load(f)
    
with open('API/scaler.pkl', 'rb') as f:
    scaler = pk.load(f)
    
with open('API/modelo_treinado.pkl', 'rb') as f:
    modelo = pk.load(f)

@app.get("/")
def hello_root():
    return {"Root": "Você está na raiz da API"}


#127.0.0.1:8000/requisicao/idade=30&salario=5000&propriedade=Alugada&ano_trabalhado=5&motivo_emprestimo=Melhoria do Lar&pontuacao=G&vl_total=10000&juros=8.8&hst_inadimplencia=0&hst_primeiro_credito=2
@app.get('/requisicao/idade={idade_cliente}&salario={salario_anual}&propriedade={propriedade}&ano_trabalhado={anos_trabalhados}&motivo_emprestimo={motivo_emp}&pontuacao={pontuacao_emp}&vl_total={valor_emp_solicitado}&juros={taxa_juros}&hst_inadimplencia={inadimplencia_anterior}&hst_primeiro_credito={anos_primeira_solc_emp}')
def montar_requisicao(idade_cliente: int,
                      salario_anual: int,
                      propriedade: str,
                      anos_trabalhados: int,
                      motivo_emp: str,
                      pontuacao_emp: str,
                      valor_emp_solicitado: int,
                      taxa_juros: float,
                      inadimplencia_anterior: int,
                      anos_primeira_solc_emp: int):
    global variaveis, dados

    variaveis = {"idade": [idade_cliente],
                 "salario": [salario_anual],
                 "propriedade": [propriedade],
                 "ano_trabalhados": [anos_trabalhados],
                 "motivo_emprestimo": [motivo_emp],
                 "pontuacao_emprestimos": [pontuacao_emp],
                 "valor_total_solicitado": [valor_emp_solicitado],
                 "taxa_juros": [taxa_juros],
                 "historico_inadimplencia": [inadimplencia_anterior],
                 "historico_primeiro_credito": [anos_primeira_solc_emp]}
    
    dados = pd.DataFrame(variaveis, index=[0])
    
    print(dados)
    return variaveis


@app.get("/previsao")
def previsao():
    global x
    
    criar_idade(dados)
    criar_valor(dados)

    df_predicao = one_hot_enc.transform(dados)
    df_predicao = pd.DataFrame(df_predicao, columns=one_hot_enc.get_feature_names_out())

    x = scaler.transform(df_predicao)
    
    previsao = modelo.predict(x)
    
    return  {"previsao": previsao[0],
             'probability_0': modelo.predict_proba(x).tolist()[0][0],
             'probability_1': modelo.predict_proba(x).tolist()[0][1]}


@app.get("/testeimg")
def teste_retorno():
    teste = modelo.decision_function(x)
    print(teste)
    return {'ok'}