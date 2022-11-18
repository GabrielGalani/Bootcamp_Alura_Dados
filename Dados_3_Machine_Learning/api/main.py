from fastapi import FastAPI

import pandas as pd

app = FastAPI()

one_hot_enc = pd.read_pickle('one_hot_encoder.pkl')
modelo = pd.read_pickle('modelo_treinado.pkl')
scaler = pd.read_pickle('scaler.pkl')

@app.get('/modelo/v1={idade_cliente}&v2={salario_anual}&v3={propriedade}&v4={anos_trabalhados}&v5={motivo_emp}&v6={pontuacao_emp}&v7={valor_emp_solicitado}&v8={taxa_juros}&v9={percentual_emp_salario}&v10={inadimplencia_anterior}&v11={anos_primeira_solc_emp}')
def previsao_modelo(idade_cliente, salario_anual, propriedade, anos_trabalhados, 
                    motivo_emp, pontuacao_emp, valor_emp_solicitado,
                    taxa_juros, percentual_emp_salario, inadimplencia_anterior, anos_primeira_solc_emp):
    
    dados = {
        'idade_cliente': [float(idade_cliente)],
        'salario_anual': [float(salario_anual)],
        'propriedade': [propriedade],
        'anos_trabalhados': [float(anos_trabalhados)],
        'motivo_emp': [motivo_emp],
        'pontuacao_emp': [pontuacao_emp],
        'valor_emp_solicitado': [float(valor_emp_solicitado)],
        'taxa_juros': [float(taxa_juros)],
        'percentual_emp_salario': [float(percentual_emp_salario)],
        'inadimplencia_anterior': [float(inadimplencia_anterior)],
        'anos_primeira_solc_emp': [float(anos_primeira_solc_emp)]
    }

    dados = pd.DataFrame(dados)

    dados = one_hot_enc.transform(dados)
    dados_transformados = pd.DataFrame(dados, columns=one_hot_enc.get_feature_names_out())

    dados_transformados = scaler.transform(dados_transformados)
    dados_transformados = pd.DataFrame(dados_transformados, columns = one_hot_enc.get_feature_names_out())
    return {'result': modelo.predict(dados_transformados)[0],
            'probability_0': modelo.predict_proba(dados_transformados).tolist()[0][0],
            'probability_1': modelo.predict_proba(dados_transformados).tolist()[0][1]}