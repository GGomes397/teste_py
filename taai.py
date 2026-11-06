# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

# -------------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS
# -------------------------------------------------------------
file_path = Path("Pesquisa- Atividade-Dia 2 (Responses).xlsx")
df = pd.read_excel(file_path)

# -------------------------------------------------------------
# 2. CLASSIFICAÇÃO DOS PERFIS (RODA DA VIDA ATUAL)
# -------------------------------------------------------------
current_cols = [c for c in df.columns if 'atualmente' in c and '[' in c]
dim_names = [c.split('[')[1].replace(']', '').strip() for c in current_cols]

X_raw = df[current_cols].values
# Reescalonamento linear 1-5 para 1-10
X_scaled = 1.0 + (X_raw - 1.0) * (9.0 / 4.0)

def classificar_perfil(row):
    mean_val = np.mean(row)
    std_val = np.std(row)
    min_val = np.min(row)
    max_val = np.max(row)
    range_val = max_val - min_val

    prof_fin = np.mean([row[1], row[3], row[4]])
    afetivo = np.mean([row[6], row[7], row[10], row[11]])
    saude_lazer = np.mean([row[0], row[8], row[9]])
    pct_abaixo_5 = np.mean(row < 5.0)

    if mean_val < 4.5 and pct_abaixo_5 >= 0.6:
        return 'Sobrevivente'
    if mean_val >= 7.0 and std_val <= 1.8 and min_val >= 5.0:
        return 'Roda Perfeita'
    if prof_fin >= 7.0 and (saude_lazer <= 4.5 or afetivo <= 4.5):
        return 'Workaholic'
    if afetivo >= 7.0 and prof_fin <= 5.5:
        return 'Família e Lar'
    if range_val >= 6.0 and std_val >= 2.2:
        return 'Montanha-Russa'

    if std_val < 1.5 and mean_val >= 5.5:
        return 'Roda Perfeita'
    elif prof_fin > afetivo + 1.5:
        return 'Workaholic'
    elif afetivo > prof_fin + 1.5:
        return 'Família e Lar'
    elif range_val >= 5.0:
        return 'Montanha-Russa'
    elif mean_val < 5.0:
        return 'Sobrevivente'
    else:
        return 'Roda Perfeita' if mean_val >= 6.0 else 'Montanha-Russa'

df['Perfil'] = [classificar_perfil(r) for r in X_scaled]

# Criação de variáveis indicadoras binárias (dummies) para cada perfil
perfis_lista = ['Roda Perfeita', 'Workaholic', 'Família e Lar', 'Montanha-Russa', 'Sobrevivente']
for perfil in perfis_lista:
    df[f'Perfil_{perfil}'] = (df['Perfil'] == perfil).astype(int)

# -------------------------------------------------------------
# 3. TRATAMENTO DAS VARIÁVEIS DE ANÁLISE
# -------------------------------------------------------------

# Projeção de Futuro (10 anos): média das notas da Roda da Vida futura
future_cols = [c for c in df.columns if '10 anos' in c and '[' in c]
if future_cols:
    df['Media_Futuro_10anos'] = df[future_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)

# Tempo de Tela (Horas de Celular)
col_tela = [c for c in df.columns if 'celular' in c.lower()]
if col_tela:
    df['Horas_Celular'] = pd.to_numeric(df[col_tela[0]], errors='coerce')

# Satisfação com o curso
col_sat = [c for c in df.columns if 'satisfação com seu curso' in c.lower()]
if col_sat:
    df['Satisfacao_Curso'] = pd.to_numeric(df[col_sat[0]], errors='coerce')

# IRA Ordinal
col_ira = [c for c in df.columns if 'ira:' in c.lower()]
if col_ira:
    ordem_ira = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    ira_cat = pd.Categorical(df[col_ira[0]].astype(str).str.strip(), categories=ordem_ira, ordered=True)
    df['IRA_Ordinal'] = ira_cat.codes + 1
    df.loc[df['IRA_Ordinal'] <= 0, 'IRA_Ordinal'] = np.nan

# Expectativa de Empregabilidade e Medo de IA (Mercado de Trabalho)
col_emp = [c for c in df.columns if '3 meses após a formatura' in c]
if col_emp:
    df['Empregabilidade'] = pd.to_numeric(df[col_emp[0]], errors='coerce')

col_ia = [c for c in df.columns if 'medo da ia' in c.lower()]
if col_ia:
    df['Medo_IA'] = pd.to_numeric(df[col_ia[0]], errors='coerce')

# -------------------------------------------------------------
# 4. CÁLCULO DA CORRELAÇÃO DE CADA PERFIL COM AS VARIÁVEIS
# -------------------------------------------------------------
variaveis_interesse = [
    'Media_Futuro_10anos',
    'Horas_Celular',
    'Satisfacao_Curso',
    'IRA_Ordinal',
    'Empregabilidade',
    'Medo_IA'
]

# Filtra apenas as colunas que efetivamente existem nos dados tratados
variaveis_validas = [v for v in variaveis_interesse if v in df.columns]

matriz_correlacao = pd.DataFrame(index=variaveis_validas, columns=perfis_lista)
matriz_pvalor = pd.DataFrame(index=variaveis_validas, columns=perfis_lista)

for var in variaveis_validas:
    for perfil in perfis_lista:
        dummy_col = f'Perfil_{perfil}'
        sub_df = df[[dummy_col, var]].dropna()
        
        if len(sub_df) > 5 and sub_df[dummy_col].nunique() > 1:
            rho, p_val = spearmanr(sub_df[dummy_col], sub_df[var])
            matriz_correlacao.loc[var, perfil] = rho
            matriz_pvalor.loc[var, perfil] = p_val
        else:
            matriz_correlacao.loc[var, perfil] = np.nan
            matriz_pvalor.loc[var, perfil] = np.nan

matriz_correlacao = matriz_correlacao.astype(float)
matriz_pvalor = matriz_pvalor.astype(float)

print("=== COEFICIENTES DE CORRELAÇÃO (SPEARMAN) ===")
print(matriz_correlacao.round(4))

print("\n=== VALORES-P (SIGNIFICÂNCIA) ===")
print(matriz_pvalor.round(4))

# -------------------------------------------------------------
# 5. GERANDO MAPA DE CALOR COM ANOTAÇÕES DE SIGNIFICÂNCIA
# -------------------------------------------------------------
# Cria rótulos com asteriscos para sinalizar significância estatística (* p < 0.05, ** p < 0.01)
rotulos = matriz_correlacao.copy().astype(str)
for var in variaveis_validas:
    for perfil in perfis_lista:
        r = matriz_correlacao.loc[var, perfil]
        p = matriz_pvalor.loc[var, perfil]
        if np.isnan(r):
            rotulos.loc[var, perfil] = "-"
        else:
            sinal = ""
            if p < 0.01:
                sinal = "**"
            elif p < 0.05:
                sinal = "*"
            rotulos.loc[var, perfil] = f"{r:.2f}{sinal}"

plt.figure(figsize=(10, 6))
sns.heatmap(
    matriz_correlacao,
    annot=rotulos,
    fmt="",
    cmap="coolwarm",
    center=0,
    vmin=-0.6,
    vmax=0.6,
    linewidths=1,
    linecolor="white",
    cbar_kws={'label': 'Coeficiente de Correlação (rho)'}
)

plt.title("Associação entre Perfis da Roda da Vida e Indicadores da Pesquisa\n(* p < 0.05, ** p < 0.01)", fontsize=12, pad=15)
plt.ylabel("Variáveis Analisadas", fontsize=11)
plt.xlabel("Perfis Classificados", fontsize=11)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()

# Salva a figura para uso na apresentação
plt.savefig("correlacao_perfis_indicadores.png", dpi=300, bbox_inches="tight")
plt.show()