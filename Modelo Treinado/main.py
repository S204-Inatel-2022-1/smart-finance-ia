import joblib
import pandas as pd  # lib para análise de dados
import os  # lib para adiministrar arquivos do projeto
from pandas_datareader import data as web
import numpy as np
from sklearn.preprocessing import StandardScaler

modelo_treinado = joblib.load("ia_carteiraacoes.joblib")

empresas = ["ABEV3", "AZUL4", "AMER3", "B3SA3", "BBSE3", "BRML3", "BBDC4", "BRAP4", "BBAS3", "BRKM5", "BRFS3", "BPAC11",
            "CRFB3", "CCRO3", "CMIG4", "CIEL3", "COGN3", "CPLE6", "CSAN3", "CPFE3", "CVCB3", "CYRE3", "ECOR3", "ELET6",
            "EMBR3", "ENBR3", "ENGI11", "ENEV3", "EGIE3", "EQTL3", "EZTC3", "FLRY3", "GGBR4", "GOAU4", "GOLL4", "NTCO3",
            "HAPV3", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "JHSF3", "KLBN11", "RENT3", "LCAM3", "LREN3", "MGLU3", "MRFG3",
            "BEEF3", "MRVE3", "MULT3", "PCAR3", "PETR4", "VBBR3", "PRIO3", "QUAL3", "RADL3", "RAIL3", "SBSP3", "SANB11",
            "CSNA3", "SULA11", "SUZB3", "TAEE11", "VIVT3", "TIMS3", "TOTS3", "UGPA3", "USIM5", "VALE3", "VIIA3",
            "WEGE3", "YDUQ3"]

fundamentos = {}
arquivos = os.listdir("balancos")  # acessando o balanço das empresas

for arquivo in arquivos:
    # pegando apenas o nome da empresa no nome do arquivo
    nome = arquivo[-9:-4]
    if "11" in nome:
        nome = arquivo[-10:-4]

    if nome in empresas:
        # pegar o balanco da empresa
        balanco = pd.read_excel(f'balancos/{arquivo}', sheet_name=0)

        # na primeira coluna colocar o título com o nome da empresa
        balanco.iloc[0, 0] = nome

        # tornando a 1ª linha em cabeçalho
        balanco.columns = balanco.iloc[0]
        balanco = balanco[1:]

        # tornar a 1ª coluna índice
        balanco = balanco.set_index(nome)

        dre = pd.read_excel(f'balancos/{arquivo}', sheet_name=1)

        # na primeira coluna colocar o título com o nome da empresa
        dre.iloc[0, 0] = nome

        # tornando a 1ª linha em cabeçalho
        dre.columns = dre.iloc[0]
        dre = dre[1:]

        # tornar a 1ª coluna índice
        dre = dre.set_index(nome)

        fundamentos[nome] = balanco.append(dre)  # adicionando no dicionário o balanço com o dre da empresa

data_inicial = "12/20/2012"
data_final = "03/31/2022"

cotacoes = {}

for empresa in empresas:
    cotacao = web.DataReader(f'{empresa}.SA', data_source='yahoo', start=data_inicial, end=data_final)  #mm/dd/yyyy
    cotacao['Empresa'] = empresa
    cotacoes[empresa] = cotacao.loc[cotacao['Empresa']==empresa, :]
    cotacoes[empresa] = cotacoes[empresa].reset_index()

for empresa in empresas:
    if cotacoes[empresa].isnull().values.any():
        # removendo as empresas das cotações e dos fundamentos
        cotacoes.pop(empresa)
        fundamentos.pop(empresa)
empresas = list(cotacoes.keys()) # lista com todos os nomes das empresas

for empresa in fundamentos:
    tabela = fundamentos[empresa].T  # trocando linha por coluna
    tabela.index = pd.to_datetime(tabela.index, format="%d/%m/%Y")  # tratando as datas

    tabela_cotacao = cotacoes[empresa].set_index("Date")  # tornando a data no índice
    tabela_cotacao = tabela_cotacao[["Adj Close"]]  # tabela cotação vai ser apenas a coluna Adj Close

    # Juntando Adj Close com cotações
    tabela = tabela.merge(tabela_cotacao, right_index=True, left_index=True)
    tabela.index.name = empresa
    fundamentos[empresa] = tabela

colunas = list(fundamentos["ABEV3"].columns)

for empresa in empresas:
    # Verificando se as empresas tem as mesmas colunas
    if set(colunas) != set(fundamentos[empresa].columns):
        fundamentos.pop(empresa) # excluindo empresa que possui coluna diferente

texto_colunas = ";".join(colunas) # unindo os nomes em um único texto

colunas_modificadas = []
for coluna in colunas:
    if colunas.count(coluna) == 2 and coluna not in colunas_modificadas: # se houver mais de uma coluna com o nome em questão
        # e ela ainda não foi tratada
        texto_colunas = texto_colunas.replace(";" + coluna + ";",";" + coluna + "_1;", 1) # substituindo nome do primeiro caso
        colunas_modificadas.append(coluna)

colunas = texto_colunas.split(';') # separando em colunas novamente

for empresa in fundamentos:
    fundamentos[empresa].columns = colunas

valores_vazios = dict.fromkeys(colunas, 0)  # inicializando dicionário

total_linhas = 0

for empresa in fundamentos:
    # contando valores vazios
    tabela = fundamentos[empresa]
    total_linhas += tabela.shape[0]

    for coluna in colunas:  # verificando a quantidade de valores vazios das colunas
        qtde_vazios = pd.isnull(tabela[coluna]).sum()
        valores_vazios[coluna] += qtde_vazios

# removendo as colunas
remover_colunas = []

for coluna in valores_vazios:
    if valores_vazios[coluna] > 50:
        remover_colunas.append(coluna)

for empresa in fundamentos:
    fundamentos[empresa] = fundamentos[empresa].drop(remover_colunas, axis=1)
    fundamentos[empresa] = fundamentos[empresa].ffill() # preenchendo o valor vazio com o valor que esta acima

data_inicial = "12/20/2012"
data_final = "03/31/2022"

df_ibov = web.DataReader('^BVSP', data_source='yahoo', start=data_inicial, end=data_final) # dataFrame do ibovespa

datas = fundamentos["ABEV3"].index
for data in datas:
    if data not in df_ibov.index:  # se a data não existir no dataFrame iremos inserir
        df_ibov.loc[data] = np.nan

df_ibov = df_ibov.sort_index()  # ordenando as datas
df_ibov = df_ibov.ffill()  # se a linha estiver vazia, preencher com o valor de cima da tabela
df_ibov = df_ibov.rename(columns={"Adj Close": "IBOV"})  # renomeando a coluna Adj Close por IBOV

for empresa in fundamentos:  # juntando IBOV na tabela de fundamentos
    fundamentos[empresa] = fundamentos[empresa].merge(df_ibov[["IBOV"]], left_index=True, right_index=True)

for empresa in fundamentos:
    fundamento = fundamentos[empresa]
    fundamento = fundamento.sort_index()  # ordenando crescentemente as datas

    for coluna in fundamento:
        if "Adj Close" in coluna or "IBOV" in coluna:  # pegar a cotação seguinte
            pass

        else:  # pegar a cotação anterior (fundamentos)

            # tratando os casos de números negativos e 0
            condicoes = [
                (fundamento[coluna].shift(1) > 0) & (fundamento[coluna] < 0),
                (fundamento[coluna].shift(1) < 0) & (fundamento[coluna] > 0),
                (fundamento[coluna].shift(1) < 0) & (fundamento[coluna] < 0),
                (fundamento[coluna].shift(1) == 0) & (fundamento[coluna] > 0),
                (fundamento[coluna].shift(1) == 0) & (fundamento[coluna] < 0),
                (fundamento[coluna].shift(1) < 0) & (fundamento[coluna] == 0),
            ]
            valores = [
                -1,
                1,
                (abs(fundamento[coluna].shift(1)) - abs(fundamento[coluna])) / abs(fundamento[coluna].shift(1)),
                1,
                -1,
                1,
            ]
            fundamento[coluna] = np.select(condicoes, valores,
                                           default=fundamento[coluna] / fundamento[coluna].shift(1) - 1)

    fundamento["Adj Close"] = fundamento["Adj Close"].shift(-1) / fundamento["Adj Close"] - 1

    fundamento["IBOV"] = fundamento["IBOV"].shift(-1) / fundamento["IBOV"] - 1

    fundamento["Resultado"] = fundamento["Adj Close"] - fundamento["IBOV"]  # cotação - IBOV

    condicoes = [
        (fundamento["Resultado"] > 0),  # comprar
        (fundamento["Resultado"] < 0) & (fundamento["Resultado"] >= -0.02),  # não comprar
        (fundamento["Resultado"] < -0.02)  # vender
    ]

    valores = [2, 1, 0]
    print("PASSEI AQUI")
    fundamento["Decisao"] = np.select(condicoes, valores)

    fundamentos[empresa] = fundamento  # atribuindo a tabela modificada no dicionário

# remover valores vazios
colunas = list(fundamentos["ABEV3"].columns)
valores_vazios = dict.fromkeys(colunas, 0)
total_linhas = 0
for empresa in fundamentos:
    tabela = fundamentos[empresa]
    total_linhas += tabela.shape[0]
    for coluna in colunas:
        qtde_vazios = pd.isnull(tabela[coluna]).sum()
        valores_vazios[coluna] += qtde_vazios

remover_colunas = []
for coluna in valores_vazios:
    if valores_vazios[coluna] > (total_linhas / 3):
        remover_colunas.append(coluna)

for empresa in fundamentos:
    fundamentos[empresa] = fundamentos[empresa].drop(remover_colunas, axis=1)
    fundamentos[empresa] = fundamentos[empresa].fillna(0) # preenchendo os valores vazios com 0

for empresa in fundamentos:
    # removendo as colunas Adj Close, IBOV, Resultado
    fundamentos[empresa] = fundamentos[empresa].drop(["Adj Close", "IBOV", "Resultado"], axis=1)

ult_tri_fundamentos = fundamentos.copy()
ult_tri_base_dados = pd.DataFrame()
lista_empresas = []

for empresa in ult_tri_fundamentos:
    ult_tri_fundamentos[empresa] = ult_tri_fundamentos[empresa][-1:]

    ult_tri_fundamentos[empresa] = ult_tri_fundamentos[empresa].reset_index(drop=True)

    ult_tri_base_dados = ult_tri_base_dados.append(ult_tri_fundamentos[empresa])

    lista_empresas.append(empresa)

'''
def ajustar_scaler(tabela_original):
    scaler = StandardScaler()

    tabela_auxiliar = tabela_original.drop("Decisao", axis=1)

    # criando a tabela ajustada
    tabela_auxiliar = pd.DataFrame(scaler.fit_transform(tabela_auxiliar), tabela_auxiliar.index,
                                   tabela_auxiliar.columns)

    tabela_auxiliar["Decisao"] = tabela_original["Decisao"]  # reinserindo a coluna de decisão

    return tabela_auxiliar
'''
top10 = ['Resultado Antes Tributação/Participações', 'Lucros/Prejuízos Acumulados', 'Obrigações Fiscais', 'Ativo Realizável a Longo Prazo', 'Receitas Financeiras', 'Fornecedores', 'Outros', 'Outras Despesas Operacionais', 'Reservas de Lucros', 'Estoques_1']

# ajustando o índice
ult_tri_base_dados = ult_tri_base_dados.reset_index(drop=True)

# pegar apenas os fundamentos mais importantes (top10)
ult_tri_base_dados = ult_tri_base_dados[top10]

# ajustando a base de dados com o scaler
#ult_tri_base_dados = ajustar_scaler(ult_tri_base_dados)

# excluindo a coluna de decisão
#ult_tri_base_dados = ult_tri_base_dados.drop("Decisao", axis=1)

# previsões
previsoes_ult_tri = modelo_treinado.predict(ult_tri_base_dados)
print(previsoes_ult_tri)

carteira = []
carteira_inicial = []

for i, empresa in enumerate(lista_empresas):
    if previsoes_ult_tri[i] == 2:  # se for para comprar
        print(empresa)

        carteira_inicial.append(1000)  # carteira inicial com R$1.000,00

        cotacao = cotacoes[empresa]
        cotacao = cotacao.set_index("Date")  # tornando a coluna data em índice

        cotacao_inicial = cotacao.loc["2021-12-30", "Adj Close"]  # cotação no dia da compra
        cotacao_final = cotacao.loc["2022-03-31", "Adj Close"]  # cotação no trimestre seguinte

        percentual = cotacao_final / cotacao_inicial

        carteira.append(1000 * percentual)
