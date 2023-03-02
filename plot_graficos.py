import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from unidecode import unidecode
from typing import Any
from processamento_dados import mean, std
estacoes = ['INVERNO', 'PRIMAVERA', 'VERÃO', 'OUTONO']


def criar_diretorio_imagens(caminho_imagem):
    try:
        # Criando diretório para salvar arquivos
        os.makedirs(f"Imagens/{caminho_imagem}")
    except FileExistsError:
        # Caso o diretório já exista
        pass


def salvar_imagens(subpasta: str, nome_imagem: str):
    """
    Salva o gráfico gerado na pasta definida com o nome definido, e caso não exista a pasta ela será criada"""
    try:
        # Criando diretório para salvar arquivos
        os.makedirs(f"Imagens/{subpasta}")
    except FileExistsError:
        # Caso o diretório já exista
        pass
    plt.savefig(f"Imagens/{subpasta}/{nome_imagem}",
                bbox_inches='tight', dpi=500)


def plot_media_desvio_padrao(dado: pd.DataFrame | pd.Series, niveis_altitude: np.ndarray, ax: plt.Axes, cor: str | None = None):
    """
    Adiciona linha de média e hachura do desvio padrão ao Axes"""
    p = ax.plot(dado['mean'], niveis_altitude, color=cor)
    # Plotando desvio padrão como preenchimento
    color = p[0].get_color()
    ax.fill_betweenx(niveis_altitude,
                     dado['mean'] + dado['std'],
                     dado['mean'] - dado['std'],
                     alpha=0.5, label='_nolegend_', color=color)


def plot_media_deteccoes_1d(media_sazonal_deteccoes: pd.DataFrame, niveis_altitude: np.ndarray, area: str, salvar=True):
    """
    Plota gráfico com o perfil vertical da média sazonal de detecções totais de aerossóis
    """
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')

    maximo_std = media_sazonal_deteccoes.loc[:, (list(media_sazonal_deteccoes.columns.unique(0)), ['std'])].max().max()
    maximo = media_sazonal_deteccoes.max().max() + maximo_std + maximo_std/10

    for estacao in estacoes:
        plot_media_desvio_padrao(
            media_sazonal_deteccoes[estacao], niveis_altitude, ax)
        # Configurações do gráfico
        ax.set_xlabel('Detecções (pixel)', fontsize=10)
        ax.set_ylabel('Altura (Km)', fontsize=10)
        ax.set_xlim(0, maximo)
        ax.set_xticks(np.round(np.linspace(0, maximo, 10)))
        ax.set_yticks(np.arange(0, 13, 2))
        ax.grid(True)

    # Adicionando legenda
    plt.legend(estacoes)
    fig.tight_layout()
    if salvar == True:
        salvar_imagens(
            'Imagens_L3', f"media_sazonal_deteccoes_aerossois_{area}.png")


def plot_deteccoes_sazonais_2d(media_sazonal_deteccoes: pd.DataFrame, niveis_altitude: np.ndarray, area: str, informacoes_variavel: dict[str, Any], salvar=True):
    """
    Plota gráfico com o perfil vertical da média sazonal de detecções por tipo de aerossol
    """
    fig, axs = plt.subplots(2, 2, figsize=(
        7, 7), facecolor='white', sharex=True, sharey=True)

    maximo = media_sazonal_deteccoes.max().max(
    ) + 2*media_sazonal_deteccoes.max().max()/10

    maximo_std = media_sazonal_deteccoes.loc[:, (list(media_sazonal_deteccoes.columns.unique(0)),
                                                 list(media_sazonal_deteccoes.columns.unique(1)),
                                                 ['std'])].max().max()
    maximo = media_sazonal_deteccoes.max().max() + maximo_std + maximo_std/10

    for estacao, ax in zip(estacoes, axs.flatten()):
        for tipo in media_sazonal_deteccoes.columns.unique(1):
            plot_media_desvio_padrao(media_sazonal_deteccoes[estacao][str(tipo)],
                                     niveis_altitude, ax, informacoes_variavel['Tipos_Aerossóis'][tipo])
        ax.set_xlabel('Detecções (pixel)', fontsize=10)
        ax.set_ylabel('Altura (Km)', fontsize=10)
        ax.set_title(estacao)
        ax.set_xlim(0, maximo)
        ax.set_xticks(np.round(np.linspace(0, maximo, 10)))
        ax.set_yticks(np.arange(0, 13, 2))
        ax.grid(True)

    fig.legend(media_sazonal_deteccoes.columns.unique(1), loc='lower left', bbox_to_anchor=(0.05, -.07),  ncol=3)
    fig.tight_layout()

    if salvar == True:
        salvar_imagens(
            'Imagens_L3', f"media_sazonal_tipos_aerossois_{area}.png")

def plot_media_movel_deteccoes_eoa(dados_deteccao: pd.DataFrame, dados_eoa: pd.DataFrame, area, salvar: bool = True):
    """
    Plot das médias móveis anuais do total de detecções da coluna e da espessura óptica 
    """
    media_movel = dados_deteccao.rolling(12, min_periods=10).agg([mean, std])
    std_movel = pd.concat([media_movel, dados_eoa.rolling(12, min_periods=10).agg([mean, std])], axis=1)[0]['std']
    std_movel.columns = ['Detecções Aerossóis', 'AOD']
    media_movel = pd.concat([media_movel, dados_eoa.rolling(12, min_periods=10).agg([mean, std])], axis=1)[0]['mean']
    media_movel.columns = ['Detecções Aerossóis', 'AOD']

    fig, ax = plt.subplots(figsize = (5,4), facecolor='white')
    # make a plot
    ax.plot(media_movel.index,
            media_movel['Detecções Aerossóis'],
            color="red")
    ax.fill_betweenx
    ax.fill_between(media_movel.index,
                    media_movel['Detecções Aerossóis'] + std_movel['Detecções Aerossóis'],
                    media_movel['Detecções Aerossóis'] - std_movel['Detecções Aerossóis'],
                     alpha=0.4, label='_nolegend_', color='red')
    # set x-axis label
    ax.set_xlabel("Data", fontsize=12)
    ax.set_xticks(media_movel.index[::12])
    ax.set_xticklabels(media_movel.index[::12], rotation=90)

    # set y-axis label
    ax.set_ylabel("Detecções",
                  color="red",
                  fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(media_movel.index, media_movel['AOD'], color="blue")
    ax2.fill_between(media_movel.index, 
                     media_movel['AOD'] + std_movel['AOD'],
                     media_movel['AOD'] - std_movel['AOD'],
                     alpha=0.4, label='_nolegend_', color='blue')
    ax2.set_ylabel("Espessura Óptica", color="blue", fontsize=12)
    if salvar:
        salvar_imagens('Imagens_L3', f"media_movel_deteccoes_vs_eoa_{area}.png")

def plot_coeficiente_extincao(media_sazonal: pd.DataFrame, informacoes_variaveis: dict, niveis_altitude: np.ndarray, area: str, salvar=True):
    """
    Plot da média sazonal do perfil vertical do coeficiente de extinção e dos valores de espessura óptica para diferentes tipos de aerossóis
    """

    fig, axs = plt.subplots(2, 2, figsize=(
        7, 7), facecolor='white', sharex=True, sharey=True)
    maximo_std = media_sazonal['Extinction_Coefficient_532_Mean'].T.loc[:,
                                                                        (list(media_sazonal['Extinction_Coefficient_532_Mean'].T.columns.unique(0)),
                                                                         ['std'])].max().max()
    maximo = media_sazonal['Extinction_Coefficient_532_Mean'].max(
    ).max() + maximo_std + maximo_std/6
    for estacao, ax in zip(estacoes, axs.flatten()):
        informacoes_ext = {variavel: valor for variavel, valor in informacoes_variaveis.items() if "Ext" in variavel}
        for tipo, info_variavel in informacoes_ext.items():

            plot_media_desvio_padrao(
                media_sazonal[tipo].T[estacao], niveis_altitude, ax, info_variavel['Cor'])
            # Configurações do gráfico
            ax.set_xlabel('Coeficiente de Extinção (km-1)', fontsize=10)
            ax.set_ylabel('Altura (Km)', fontsize=10)
            ax.set_xlim(0, maximo)
            ax.set_xticks(np.round(np.linspace(0, maximo, 6), 2))
            ax.set_ylim(0, 8)
            ax.set_yticks(np.arange(0, 9, 1))
            ax.grid(True)
            ax.set_title(estacao)

        # Adiconando caixa de texto com valores de espessura óptica nos subplots
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        # Criando texto com valores de espessura óptica para cada variável
        valores = '\n'.join([f"{info['Tradução'].split(' - ')[1]}: \u00B1{format(media_sazonal['AOD_Mean'].T[estacao]['mean'][0].round(3), '.3f')}"
                             for variavel, info in informacoes_variaveis.items() if "AOD" in variavel])
        texto = f"EOA:\n{valores}"
        ax.text(0.25, 0.95, texto, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=props)
    # Adicionando legenda
    legenda = [info['Tradução'].split(
        ' - ')[1] for variavel, info in informacoes_variaveis.items() if 'Ext' in variavel]
    lgd = fig.legend(legenda, loc='lower left', bbox_to_anchor=(
        0.115, -.05), ncol=4, borderaxespad=0.)
    fig.tight_layout()
    criar_diretorio_imagens('Imagens_L3/')
    if salvar:
        plt.savefig(f"Imagens/Imagens_L3/media_sazonal_coef_extincao_espessura_{area}.png", bbox_extra_artists=(
            lgd,), bbox_inches='tight', dpi=500)


def plot_multiplo_perfil_deteccao(perfil_media_sazonal: dict[str, Any], info_variaveis: dict[str, Any], niveis_altitude: np.ndarray, salvar: bool = True):
    """
    Plota perfil de detecções de aerossóis para cada região em subplots"""
        
    fig, axs = plt.subplots(3,4,figsize=(7,7), facecolor='white', sharex=True, sharey=True)
    for area, ax in zip(perfil_media_sazonal.keys(), axs.flatten()):
        perfil = perfil_media_sazonal[area]['Samples_Aerosol_Detected_Accepted'].T
        for estacao in perfil.columns.unique(0):
            plot_media_desvio_padrao(perfil[estacao], niveis_altitude, ax, info_variaveis['Cores'][estacao])
            # Configurações do gráfico
            # ax.set_xlim(0, maximo)
            ax.set_xticks(np.round(np.arange(0, 400, 100)))
            ax.set_yticks(np.arange(0, 13, 2))
            ax.set_title(f"{' '.join(area.split('_')[2:])}", fontsize = 10)
            ax.grid(True)
    # Configurando o nome comum cos eixos
    fig.text(0.5, 0, 'Detecções (pixel)', ha='center', va='center', fontsize = 10)
    fig.text(0, 0.5, 'Altura (Km)', ha='center', va='center', rotation='vertical', fontsize = 10)
    # Adicionando legenda
    fig.legend(perfil.columns.unique(0), loc='lower center', bbox_to_anchor=(.5,1), ncol=4, borderaxespad=0., fontsize = 10)
    fig.tight_layout()
    if salvar == True:
        salvar_imagens('Imagens_L3', f"perfil_deteccoes_sazonais_multiplas_areas.png")

def plots_multiplos_perfil_ce_eoa(perfil_media_sazonal: dict[str, Any], info_variaveis: dict[str, Any], niveis_altitude: np.ndarray, salvar: bool = True):
    """
    Plota gráfico do perfil sazonal de detecções do coeficiente de extinção e valores de espessura óptica dos aerossóis
    em 4 figuras separadas, cada figura com os gráficos das 4 estações em linhas e 3 areas em colunas
    """
    for i in [0,3,6,9]:
        numero_areas = []
        fig, axs = plt.subplots(4, 3, figsize=(8, 9), facecolor='white', sharex=True, sharey=True)
        for estacao, axs_interno in zip(estacoes, axs):
            # media_sazonal = perfil_media_sazonal[area]
            
            for (area, media_sazonal), ax in zip(list(perfil_media_sazonal.items())[i:i+3], axs_interno):
                numero_areas.append(area.split('_')[1])
                informacoes_ext = {variavel: valor for variavel, valor in info_variaveis.items() if "Ext" in variavel}
                for tipo, info_variavel in informacoes_ext.items():
                    plot_media_desvio_padrao(media_sazonal[tipo].T[estacao], niveis_altitude, ax, info_variavel['Cor'])
                # Configurações do gráfico
                ax.set_xlim(0,1.2)
                ax.set_xticks(np.arange(0,1.3,.3))
                ax.set_ylim(-0.5, 8)
                ax.set_yticks(np.arange(0, 9, 2))
                ax.grid(True)
                ax.set_title(f"{' '.join(area.split('_')[2:])}")

                # Adiconando caixa de texto com valores de espessura óptica nos subplots
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                # Criando texto com valores de espessura óptica para cada variável
                valores = '\n'.join([f"{info['Tradução'].split(' - ')[1]}: \u00B1{format(media_sazonal[variavel].T[estacao]['mean'][0].round(3), '.3f')}"
                                    for variavel, info in info_variaveis.items() if "AOD" in variavel])
                texto = f"EOA:\n{valores}"
                ax.text(0.25, 0.95, texto, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
            
            axs_interno[0].set_ylabel(estacao, fontsize=10)
            
        # Adicionando labels para todos os subplots
        fig.text(0.5, 0, 'Coef. Ext. (km^-1)', ha='center', va='center', fontsize=10)
        nome_eixo = fig.text(0, 0.5, 'Altura (km)', ha='center', va='center',
                rotation='vertical', fontsize=10)
        # Adicionando legenda
        legenda = [info['Tradução'].split(' - ')[1] for variavel, info in info_variaveis.items() if 'Ext' in variavel]
        lgd = fig.legend(legenda, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, borderaxespad=0.)
        fig.tight_layout()

        criar_diretorio_imagens('Imagens_L3/')
        if salvar:
            plt.savefig(f"Imagens/Imagens_L3/perfil_sazonal_ce_eao_areas_{numero_areas[0]}_a_{numero_areas[-1]}.png", 
                        bbox_extra_artists=(lgd,nome_eixo), bbox_inches='tight', dpi=500)
            

def plot_multiplo_media_movel_tendencia(dados: dict, variavel: str, info_variaveis: dict[str, dict], salvar: bool = True):
    """
    Plota média movel e desvio padrão anual da coluna
    """
    fig, axs = plt.subplots(4, 3, figsize=(7, 8), facecolor='white', sharex=True, sharey=True, squeeze=True)
    for area, ax in zip(dados.keys(), axs.flatten()):
        deteccoes = dados[area][variavel].copy()
        medias_moveis = deteccoes.rolling(12, min_periods=10).agg([mean, std])

        # Plot média móvel e desvio padrão
        ax.plot(medias_moveis.index,
                medias_moveis[0]['mean'] , color="royalblue")
        ax.fill_between(medias_moveis.index, (medias_moveis[0]['mean'] + medias_moveis[0]['std']) ,
                        (medias_moveis[0]['mean'] - medias_moveis[0]['std']) , alpha=0.4, label='_nolegend_',
                        color='royalblue')

        # Plot linha de tendência
        x = np.arange(len(deteccoes))
        y = deteccoes[0]
        idx = np.isfinite(x) & np.isfinite(y)
        z = np.polyfit(x[idx], y[idx], 1)
        p = np.poly1d(z)
        ax.plot(deteccoes.index, p(x), label='Tendência', color='royalblue', linestyle='-', linewidth=.8)

        ax.set_xticks(medias_moveis.index[::24])
        ax.set_xticklabels(medias_moveis.index[::24], rotation=90, fontsize=10)
        ax.set_title(f"{' '.join(area.split('_')[2:])}", fontsize=10)

        ax.grid(True)

    fig.text(0.5, 0, 'Meses', ha='center', va='center', fontsize=10)
    fig.text(0, 0.5, info_variaveis[variavel]['Nome_eixo'], ha='center', va='center',
            rotation='vertical', fontsize=10)

    fig.tight_layout()
    if salvar == True: 
        salvar_imagens('Imagens_L3', 
                       f"media_movel_tendencia_{unidecode('_'.join(info_variaveis[variavel]['Tradução'].lower().split(' - ')[0].split(' ')))}_multiplas_areas.png")


def plot_media_movel_deteccoes_2d(dados_deteccao: pd.DataFrame, info_variavel: dict[str, Any], area: str, salvar: bool = True):
    """
    Plot da media movel das detecções em toda a coluna de cada tipo de aerossol para area individual
    """
    media_movel = dados_deteccao.rolling(12, min_periods=10).agg([mean, std])
    fig, ax = plt.subplots(figsize=(6,5), facecolor='white')
    for tipo in dados_deteccao.columns.unique(0):
        cor = info_variavel["Tipos_Aerossóis"][tipo]
        # Plot média móvel e desvio padrão
        ax.plot(media_movel.index, media_movel[tipo]['mean'], color=info_variavel['Tipos_Aerossóis'][tipo])
        ax.fill_between(media_movel.index, 
                        (media_movel[tipo]['mean'] + media_movel[tipo]['std']),
                        (media_movel[tipo]['mean'] - media_movel[tipo]['std']), alpha=0.4, label='_nolegend_',
                        color=info_variavel['Tipos_Aerossóis'][tipo])
        # Plot linha de tendência
        x = np.arange(len(media_movel[tipo]))
        y = media_movel[tipo]['mean']
        idx = np.isfinite(x) & np.isfinite(y)
        z = np.polyfit(x[idx], y[idx], 1)
        p = np.poly1d(z)
        ax.plot(media_movel[tipo].index, p(x), label='Tendência', color=cor, linestyle='-')
    ax.set_xlabel('Data', fontsize=10)
    ax.set_ylabel('Detecções (pixel)', fontsize=10)
    ax.set_xticks(media_movel.index[::12])
    ax.set_xticklabels(media_movel.index[::12], rotation=45)
    # ax.set_yticks(np.arange(0, 13, 2))
    ax.grid(True)
    lgd = fig.legend(media_movel.columns.unique(0), loc='upper center', 
            bbox_to_anchor=(0.5, 0), ncol=3, borderaxespad=0.)
    
    fig.tight_layout()

    criar_diretorio_imagens('Imagens_L3/')
    if salvar:
        plt.savefig(f"Imagens/Imagens_L3/media_movel_deteccoes_coluna_por_tipo_{area}.png", 
                    bbox_extra_artists=(lgd), bbox_inches='tight', dpi=500)