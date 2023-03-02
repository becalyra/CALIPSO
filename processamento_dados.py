from pyhdf.SD import SD, SDC
import pandas as pd
import numpy as np
from typing import Any

# Função que retorna dados das flags organizadas em arrays de 3 dimensões (lat x perfil x altitude)

estacoes = ['INVERNO', 'PRIMAVERA', 'VERÃO', 'OUTONO']


def reordenando_array(dado: np.ndarray, latitudes: np.ndarray) -> list[np.ndarray]:
    """
    Retorna lista organizada por faixa de altitude (0-8.2 km, 8.2- 20.2 km e 20.2-30.1 km)
    contendo arrays de três dimensões (N x perfis x altitude), onde N representa cada pedaço do espaço de 5km
    """
    # Pega tamanho do array que corresponde aos pontos no espaço
    N = latitudes.shape[0]
    # A faixa de altitude de 0-8.2 km de altitude é separada em 15 perfis e 290 alturas
    faixa_altitude = [np.flip(np.reshape(dado[:, 1165:], (N, 15, 290)), 2)]
    # A faixa de altitude de 8.2-20.2 km é separada em 5 perfis e 200 alturas
    faixa_altitude.append(
        np.flip(np.reshape(dado[:, 165:1165], (N, 5, 200)), 2))
    # # A faixa de altitude de 8.2-20.2 km é separada em 3 perfis e 55 alturas
    # faixa_altitude.append(np.flip(np.reshape(dado[:,:165], (N, 3, 55))))
    return faixa_altitude


def decoficando_bits(flags: np.ndarray, latitudes: np.ndarray) -> list:
    """
    Extrai tipo dos elementos e subtipo dos aerossóis, faz controle de qualidade e reformula array em latitude, 
    perfil e altitude de acordo com a resolução da faixa de altitude
    """
    # Extraindo elementos dos bits e faz controle de qualidade pelas flags de cq
    tipos_cq = (((flags >> 3) & 3) >= 2).astype('int')
    # lista com tipo do elemento dos 3 primeiros bits da flag extraido usando mascaramento de bits
    tipos = [(flags & 7) * tipos_cq]

    # Extrai subtipos aerossóis troposféricos e aerossóis estratosféricos dos bits e faz controle de qualidade pelas flags de cq
    subtipos_cq = ((flags >> 12) & 1)
    # extrai o subtipo do elementos dos 3 primeiros bits a esquerda da posição 9
    subtipos = ((flags >> 9) & 7) * subtipos_cq

    for tipo in range(3, 5):
        # array de boleano indicando onde o tipo das flags é igual ao tipo do loop
        tmask = (tipos[0] == tipo)
        # array de boleano indicando onde o subtipo das flags é diferente de 0
        temp1 = (subtipos != 0)
        # array de boleano indicando onde o tipo das flags é igual ao tipo do loop e o subtipo é diferente de zero
        temp2 = (temp1 & tmask)
        # array que mantém apenas valores dos subtipos onde onde o tipo das flags é igual ao tipo do loop e o subtipo é diferente de zero
        subtipo = subtipos * temp2
        # adiciona lista tipos um array contendo os valores dos subtipos classificados de acordo com os tipos: nevem, aerossol troposférico e aerossol estratosférico respectivamente
        tipos.append(subtipo)

    # Adiciona a cada classificação os valores de tipos e subtipos em array de 3 dimensões (lat x perfil x altitude)
    return [reordenando_array(tipo, latitudes) for tipo in tipos]


def idx_coord(lista_coordenadas: np.ndarray, coordenada: float) -> np.int64:
    """ 
    Função que retorna indice do valor da lista mais proximo ao valor passado 
    """
    return (np.abs(lista_coordenadas - (coordenada))).argmin()


def selecao_area(dado: SD, coordenadas: dict[str, list[float]], dataset: str) -> np.ndarray:
    """
    Retorna um array com corte de área a partir do dataset escolhido do HDF~
    """
    # Cria variaveis de lat e lon
    latitudes = dado.select('Latitude_Midpoint').get()[0]
    longitudes = dado.select('Longitude_Midpoint').get()[0]

    # Selecionando índices correspondentes aos pontos mais proximos ao intervalo dado
    idx_n_lat, idx_s_lat = (idx_coord(latitudes, coordenadas['Latitudes'][0]),
                            idx_coord(latitudes, coordenadas['Latitudes'][1]))
    idx_e_lon, idx_w_lon = (idx_coord(longitudes, coordenadas['Longitudes'][0]),
                            idx_coord(longitudes, coordenadas['Longitudes'][1]))

    return dado.select(dataset)[int(idx_s_lat):int(idx_n_lat) + 1, int(idx_e_lon):int(idx_w_lon) + 1]


def controle_qualidade(dado: np.ndarray, limites_deteccao: list) -> np.ndarray:
    """
    Transforma em np.nan os valores com erro e fora dos limites de detecção
    """
    # Transformando valores do array em float
    dado = dado.astype('float')
    # Substituindo valores com erro por np.nan
    dado[dado == -9999] = np.nan
    # Substituindo valores fora dos limites de detecção para cada variável por np.nan
    dado[dado < limites_deteccao[0]] = np.nan
    dado[dado > limites_deteccao[1]] = np.nan
    return dado


def porcentagem_valida(dado: np.ndarray) -> float:
    """
    Retorna a porcentagem de dados no array que não são nulos
    """
    return (dado.size - np.count_nonzero(np.isnan(dado)))/dado.size


def coluna_estacao_ano(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona uma coluna contento entação do ano e ano no dataframe a partir do índice 
    """
    dataframe['Estacão_ano'] = [f"INVERNO {x[:4]}" if 6 <= int(x[-2:]) <= 8
                                else f"PRIMAVERA {x[:4]}" if 9 <= int(x[-2:]) <= 11
                                else f"VERÃO {int(x[:4]) + 1}" if x[-2:] == '12'
                                else f"VERÃO {x[:4]}" if int(x[-2:]) <= 2
                                else f"OUTONO {x[:4]}" for x in dataframe.index.get_level_values(0)]
    return dataframe


def convert_dicionario_2d_dataframe(dicionario_entrada: dict[str, np.ndarray], info_variavel: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Transforma dicionário contendo os dados de detecções de aerossóis por tipo em forma de array para dataframe
    """
    dicionario_dataframes = {chave: pd.DataFrame(valor, columns=info_variavel['Tipos_Aerossóis'].keys())
                             for chave, valor in dicionario_entrada.items()}
    return dicionario_dataframes


def convert_dicionario_1d_dataframe(dado: dict) -> pd.DataFrame:
    """
    Tranforma dicionário com chaves correspondendo aos meses em dataframe com datas nos índices
    """
    return pd.DataFrame.from_dict(dado, orient='index')


def preenchendo_dados(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche valores de meses faltantes com NaN
    """
    dataframe.index = pd.DatetimeIndex(dataframe.index)
    indice = pd.date_range(list(dataframe.index)[0], list(dataframe.index)[-1], freq='M').to_period('m')
    dataframe.index = dataframe.index.to_period('m')
    dataframe = dataframe.reindex(indice, fill_value=np.nan)
    dataframe.index = dataframe.index.to_series().astype(str)
    return dataframe


def calc_media_espacial(dado: np.ndarray) -> np.ndarray:
    """
    Retorna a média para toda a área  a partir de array em que as duas primeiras dimensões correspondem a latitude e longitude
    """
    # Fazendo a média em relação as duas primeiras dimensões
    return np.nanmean(np.nanmean(dado, axis=0), axis=0)


def calc_media_sazonal_anual_1d(dicionario_entrada: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Retorna dataframe com média para cada estação de cada ano
    """
    dataframe = convert_dicionario_1d_dataframe(dicionario_entrada)
    # Cria coluna com estacão e ano de cada linha
    dataframe = coluna_estacao_ano(dataframe)
    # Cria um novo nível de índice a partir da coluna "Estação_ano"
    dataframe = dataframe.set_index('Estacão_ano')

    # Verificando quais estações de cada ano contam com os 3 meses para compor a média sazonal do ano
    unique, counts = np.unique(dataframe.index, return_counts=True)
    meses_est = dict(zip(unique, counts))
    excluir = [mes for mes in meses_est.keys() if meses_est[mes] < 3]
    # Removendo dados das estações que não tem 3 meses para a média
    dataframe = dataframe.drop(excluir, axis=0)
    return dataframe.groupby(level=0).mean()


def calc_media_sazonal_anual_2d(dicionario_entrada: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Retorna dataframe com média para cada estação de cada ano
    """
    # Concatena valores de cada mês salvos no dicionário
    dataframe = pd.concat(dicionario_entrada, axis=1).T
    # Cria coluna com estacão e ano de cada linha
    dataframe = coluna_estacao_ano(dataframe)
    # Cria um novo nível de índice a partir da coluna "Estação_ano"
    dataframe = dataframe.set_index('Estacão_ano', append=True)

    # Excluir estações dos anos que não tem 3 meses para compor a média sazonal
    unique, counts = np.unique(dataframe.groupby(
        level=[0, 2]).count().index.get_level_values(1), return_counts=True)
    meses_est = dict(zip(unique, counts))
    excluir = [mes for mes in meses_est.keys() if meses_est[mes] < 3]
    dataframe = dataframe.drop(excluir, level=2, axis=0)
    # Fazendo a média sazonal para cada ano
    dataframe = dataframe.groupby(level=[2, 1]).mean()
    return dataframe.loc[(dataframe != 0).any(axis=1)]

def lista_filtrada(linha):
    """
    Filtra valores fora dos limites do triplo do desvio padrãoe transforma em np.nan
    """    
    lim_max = np.nanmean(linha) + 3 * np.nanstd(linha)
    lim_min = np.nanmean(linha) - 3 * np.nanstd(linha)
    return np.where((linha>=lim_min) & (linha<= lim_max), linha, np.nan)
def mean(linha):
    """
    Calcula a média usando a os valores filtrados
    """    
    if all(np.isnan(linha)):
        return np.nan
    return np.nanmean(lista_filtrada(linha))
def std(linha):
    """
    Calcula o desvio padrão usando a os valores filtrados
    """    
    if all(np.isnan(linha)):
        return np.nan
    return np.nanstd(lista_filtrada(linha))

def calc_media_sazonal_1d(dataframe: pd.DataFrame) -> pd.DataFrame | pd.Series:
    """
    Retorna dataframe com a média sazonal a partir de um dataframe que tenha os valores para estação de cada ano
    """
    return dataframe.groupby([dataframe.index.get_level_values(0).str[:-5]]).agg([mean, std]).stack()


def calc_media_sazonal_2d(dataframe: pd.DataFrame) -> pd.DataFrame | pd.Series:
    """
    Retorna dataframe com a média sazonal a partir de um dataframe que tenha os valores para estação de cada ano
    """
    media = dataframe.groupby([dataframe.index.get_level_values(0).str[:-5],
                               dataframe.index.get_level_values(1)]).agg([mean, std]).stack()
    return media.loc[(media != 0).any(axis=1)]


def calc_anomalia_padronizada_sazonal_anual_1d(media_sazonal_anual: pd.DataFrame, media_sazonal: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula anomalia padronizada da média sazonal anual em relação a média sazonal para todo o período
    """
    media_sazonal = media_sazonal.reorder_levels([1, 0], axis=1)
    desvio_padrao_sazonal = media_sazonal['std']
    media_sazonal = media_sazonal['mean']
    dicionario_anomalias = {estacao:
                            (media_sazonal_anual[[col for col in media_sazonal_anual.columns if estacao in col]].sub(media_sazonal[estacao], axis=0)
                             ).div(desvio_padrao_sazonal[estacao], axis=0)
                            for estacao in estacoes}
    return pd.concat(dicionario_anomalias.values(), axis=1)


def calc_anomalia_padronizada_sazonal_anual_2d(media_sazonal_anual: pd.DataFrame, media_sazonal: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula anomalia padronizada da média sazonal anual em relação a média sazonal para todo o período
    """
    dicionario_anomalia = {}
    media_sazonal = media_sazonal.copy().reorder_levels([2, 0, 1], axis=1)
    desvio_padrao_sazonal = media_sazonal['std']
    media_sazonal = media_sazonal['mean']
    for estacao in estacoes:
        dicionario_anomalia[estacao] = {coluna: (media_sazonal_anual[coluna].sub(media_sazonal[estacao], axis=0) / desvio_padrao_sazonal[estacao])
                                        for coluna in [col for col in media_sazonal_anual.columns.unique(level=0) if estacao in col]}

        dicionario_anomalia[estacao] = pd.concat(
            dicionario_anomalia[estacao], axis=1)

    return pd.concat(dicionario_anomalia, axis=1)


def calc_soma_deteccoes_coluna(dado: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Calcula a soma de detecções para toda a coluna a partir de dados de perfil (1d ou 2d)
    """
    return {mes: deteccoes.sum() for mes, deteccoes in dado.items()}


def calc_media_movel_coluna(dado: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a média movel anual de detecções a partir de dados de médias mensais dos perfis de detecções por tipo de aerossóis
    """
    dataframe = preenchendo_dados(dado)
    dataframe = dataframe.rolling(12, min_periods=10).agg(['mean', 'std'])
    return dataframe.loc[:, (dataframe**2).sum() != 0] 


def calc_media_movel_anomalia_padronizada(dado_media_movel: pd.DataFrame, dado_mensal) -> pd.DataFrame:
    """
    Calcula anomalia padronizada da média movel anual
    """
    return dado_media_movel.sub(dado_mensal.mean())
