# CALIPSO
### Scripts usados para análises produzidas para a pesquisa de iniciação científica desenvolvida no Laboratório de Aplicações de Satélites Ambientais do Departamento de Meteorologia da UFRJ. São utilizados dados do instrumento lidar a bordo do satélite CALIPSO para analisar o comportamento dos aerossóis. 
### Os scripts são construídos utilizando Python. É necessária a instalação das bibliotecas pyhdf, matplotlib, pandas e numpy.

## Produto de Vertical Feature Mask de nível 2 (versão 4.20)

Descreve a distribuição vertical e horizontal de camadas de nuvens e aerossóis observadas pelo lidar CALIPSO. As flags de classificação são armazenadas como inteiros de 16 bits que descrevem diferentes aspectos dos elementos detectados na atmosfera.
A partir das flags de classificação são gerados gráficos do perfil vertical da frequência de detecção de cada tipo de aerossol encontrado.
https://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_summaries/vfm/index_v420.php#feature_classification_flags

## Produto de Tropospheric Aerosol Profile Monthly Product de nível 3 (versão 4.20)

Esse produto relata os perfis médios mensais das propriedades ópticas do aerossol em uma grade espacial uniforme. É feito um controle de qualidade dos dados de nível 2 do qual esse produto é derivado e somente depois é calculada a média mensal.
Desse produto são utilizadas as informações do perfil do coeficiente de extinção, espessura óptica, perfil dos tipos de aerossóis detectados e perfil do total de aerossóis detectados. São criados gráficos sazonais com os perfis.
https://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_summaries/l3/cal_lid_l3_tropospheric_apro_v4-20_desc.php
