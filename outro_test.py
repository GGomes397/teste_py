# ============================================================
# ANÁLISE EXPLORATÓRIA: EXPERIÊNCIA ACADÊMICA E MERCADO DE TRABALHO
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import spearmanr, chi2_contingency, fisher_exact

# ------------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS
# ------------------------------------------------------------

diretorio = Path(__file__).resolve().parent

file_path = diretorio / "Pesquisa- Atividade-Dia 2 (Responses).xlsx"

df = pd.read_excel(file_path)

print("Dimensões da base:", df.shape)

print("\nPrimeiras observações:")
print(df.head())

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid")

# Cria uma pasta para salvar os resultados
pasta_resultados = diretorio / "resultados_analise"

pasta_resultados.mkdir(exist_ok=True)

# ============================================================
# 2. IDENTIFICAÇÃO DAS VARIÁVEIS
# ============================================================

colunas = {
    "satisfacao_curso":
        "de 1 a 10, Qual seu nível de satisfação com seu curso (faculdade)",

    "facilidade_emprego":
        "(PG1) Na escala de 1 (discordo totalmente) a 5 (concordo totalmente), quanto você concorda com a seguinte afirmação: Vai ser fácil conseguir um emprego até 3 meses após a formatura.",

    "medo_ia":
        "(PG2) Na escala de 1 (Nenhum medo) a 5 (terror e desespero, socorro), quanto você tem medo da IA te substituir totalmente no trabalho?",

    "teoria_vs_pratica":
        "(PG3) Na escala de 1 (discordo totalmente) a 5 (concordo totalmente), quanto você concorda com a seguinte afirmação: É mais importante ter boas notas e dominar o conteúdo teórico do que ter experiência prática em projetos",

    "expectativa_salarial_faixa":
        "(PG4) Assinale qual alternativa melhor corresponde à sua expectativa salarial para o primeiro emprego. Obs: Quanto você acredita conseguir, não quanto você quer.",

    "impacto_ia":
        "(PG5) Na escala de 1 (piorou muito) a 5 (melhorou muito) e sendo 3 (não mudou), como você acha que o seu rendimento se alterou com o uso de IA?",

    "faixa_ira":
        "Qual seu IRA:"
}

# Verifica se todas as colunas foram encontradas
for nome, coluna in colunas.items():

    if coluna not in df.columns:
        print(f"Coluna não encontrada: {nome}")
        print(f"Nome esperado: {coluna}")

    else:
        print(f"OK: {nome}")

for coluna in df.columns:
    print(repr(coluna))

# ============================================================
# 3. SATISFAÇÃO COM O CURSO E EMPREGABILIDADE
# ============================================================

satisfacao = pd.to_numeric(
    df[colunas["satisfacao_curso"]],
    errors="coerce"
)

empregabilidade = pd.to_numeric(
    df[colunas["facilidade_emprego"]],
    errors="coerce"
)

# Junta as variáveis e remove observações incompletas
dados_1 = pd.DataFrame({
    "Satisfação com o curso": satisfacao,
    "Facilidade de conseguir emprego": empregabilidade
}).dropna()

# Estatísticas descritivas
print("\n=== ANÁLISE 1: ESTATÍSTICAS DESCRITIVAS ===")

print(dados_1.describe())

# Correlação de Spearman
rho, p_valor = spearmanr(
    dados_1["Satisfação com o curso"],
    dados_1["Facilidade de conseguir emprego"]
)

print("\n=== CORRELAÇÃO DE SPEARMAN ===")
print(f"Número de observações: {len(dados_1)}")
print(f"Coeficiente de Spearman: {rho:.4f}")
print(f"Valor-p: {p_valor:.4f}")

if p_valor < 0.05:

    print("Resultado: evidência de associação monotônica.")

else:

    print("Resultado: não foi encontrada evidência estatística suficiente de associação monotônica.")

plt.figure(figsize=(9, 6))

sns.regplot(
    data=dados_1,
    x="Satisfação com o curso",
    y="Facilidade de conseguir emprego",
    x_jitter=0.12,
    y_jitter=0.08,
    fit_reg=False,
    scatter_kws={"alpha": 0.7, "s": 70}
)

plt.title(
    "Satisfação acadêmica e percepção de empregabilidade"
)

plt.xlabel("Satisfação com o curso (1 a 10)")
plt.ylabel("Facilidade percebida de conseguir emprego (1 a 5)")

plt.tight_layout()

plt.savefig(
    pasta_resultados / "01_satisfacao_empregabilidade.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 4. TEORIA VS. PRÁTICA E EXPECTATIVA SALARIAL
# ============================================================

dados_2 = pd.DataFrame({
    "Teoria_vs_pratica": pd.to_numeric(
        df[colunas["teoria_vs_pratica"]],
        errors="coerce"
    ),

    "Expectativa_salarial": df[
        colunas["expectativa_salarial_faixa"]
    ]
})

dados_2 = dados_2.dropna()

# Frequências absolutas
tabela_2 = pd.crosstab(
    dados_2["Teoria_vs_pratica"],
    dados_2["Expectativa_salarial"]
)

print("\n=== TABELA DE CONTINGÊNCIA ===")
print(tabela_2)

# Percentuais dentro de cada nível de concordância
percentuais_2 = pd.crosstab(
    dados_2["Teoria_vs_pratica"],
    dados_2["Expectativa_salarial"],
    normalize="index"
) * 100

print("\n=== PERCENTUAIS POR NÍVEL DE CONCORDÂNCIA ===")
print(percentuais_2.round(1))

# ============================================================
# TESTE QUI-QUADRADO DE INDEPENDÊNCIA
# ============================================================

chi2, p_valor, gl, esperadas = chi2_contingency(
    tabela_2
)

print("\n=== TESTE QUI-QUADRADO ===")

print(f"Estatística qui-quadrado: {chi2:.4f}")
print(f"Graus de liberdade: {gl}")
print(f"Valor-p: {p_valor:.4f}")

esperadas_df = pd.DataFrame(
    esperadas,
    index=tabela_2.index,
    columns=tabela_2.columns
)

print("\nFrequências esperadas:")
print(esperadas_df.round(2))

proporcao_baixas = np.mean(esperadas < 5)

print(
    "\nProporção de células com frequência esperada < 5:",
    f"{proporcao_baixas:.1%}"
)

if proporcao_baixas > 0.20:

    print(
        "ATENÇÃO: mais de 20% das células têm frequência esperada "
        "inferior a 5. A aproximação qui-quadrado pode ser inadequada."
    )

if p_valor < 0.05:

    print("Resultado: evidência de associação entre as variáveis.")

else:

    print("Resultado: não foi encontrada evidência estatística suficiente de associação.")

# ============================================================
# 5. PERCEPÇÃO SOBRE A INTELIGÊNCIA ARTIFICIAL
# ============================================================

dados_3 = pd.DataFrame({
    "Medo_IA": pd.to_numeric(
        df[colunas["medo_ia"]],
        errors="coerce"
    ),

    "Impacto_IA": pd.to_numeric(
        df[colunas["impacto_ia"]],
        errors="coerce"
    ),

    "Facilidade_emprego": pd.to_numeric(
        df[colunas["facilidade_emprego"]],
        errors="coerce"
    )
})

print("\n=== ESTATÍSTICAS DESCRITIVAS SOBRE IA ===")

print(dados_3.describe())

print("\n=== DISTRIBUIÇÃO DAS RESPOSTAS ===")

for variavel in dados_3.columns:

    print(f"\n{variavel}")

    print(
        dados_3[variavel]
        .value_counts()
        .sort_index()
    )

# ============================================================
# GRÁFICOS DE DISTRIBUIÇÃO
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 5)
)

variaveis = [
    ("Medo_IA", "Medo de substituição pela IA"),
    ("Impacto_IA", "Impacto percebido da IA no rendimento"),
    ("Facilidade_emprego", "Confiança na empregabilidade")
]

for ax, (variavel, titulo) in zip(axes, variaveis):

    frequencias = (
        dados_3[variavel]
        .value_counts()
        .sort_index()
    )

    ax.bar(
        frequencias.index.astype(str),
        frequencias.values
    )

    ax.set_title(titulo)
    ax.set_xlabel("Resposta")
    ax.set_ylabel("Número de participantes")

plt.tight_layout()

plt.savefig(
    pasta_resultados / "03_distribuicao_percepcao_ia.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# ============================================================
# CORRELAÇÃO ENTRE MEDO DA IA E IMPACTO NO RENDIMENTO
# ============================================================

dados_ia = dados_3[
    ["Medo_IA", "Impacto_IA"]
].dropna()

rho_ia, p_ia = spearmanr(
    dados_ia["Medo_IA"],
    dados_ia["Impacto_IA"]
)

print("\n=== MEDO DA IA VS. IMPACTO NO RENDIMENTO ===")

print(f"Observações utilizadas: {len(dados_ia)}")
print(f"Correlação de Spearman: {rho_ia:.4f}")
print(f"Valor-p: {p_ia:.4f}")

if p_ia < 0.05:

    print("Resultado: evidência de associação monotônica.")

else:

    print("Resultado: não foi encontrada evidência estatística suficiente de associação monotônica.")

# ============================================================
# MEDO DA IA VS. EMPREGABILIDADE
# ============================================================

dados_ia_emprego = dados_3[
    ["Medo_IA", "Facilidade_emprego"]
].dropna()

rho_emprego, p_emprego = spearmanr(
    dados_ia_emprego["Medo_IA"],
    dados_ia_emprego["Facilidade_emprego"]
)

print("\n=== MEDO DA IA VS. EMPREGABILIDADE ===")

print(f"Observações utilizadas: {len(dados_ia_emprego)}")
print(f"Correlação de Spearman: {rho_emprego:.4f}")
print(f"Valor-p: {p_emprego:.4f}")

print(df[colunas["faixa_ira"]].value_counts(dropna=False))

# ============================================================
# FAIXA DO IRA VS. EMPREGABILIDADE
# ============================================================

ordem_ira = [
    "0-20",
    "20-40",
    "40-60",
    "60-80",
    "80-100"
]

dados_ira = pd.DataFrame({
    "IRA": df[colunas["faixa_ira"]].astype("string").str.strip(),

    "Empregabilidade": pd.to_numeric(
        df[colunas["facilidade_emprego"]],
        errors="coerce"
    )
}).dropna()

# Mantém somente as categorias conhecidas
dados_ira = dados_ira[
    dados_ira["IRA"].isin(ordem_ira)
].copy()

dados_ira["IRA"] = pd.Categorical(
    dados_ira["IRA"],
    categories=ordem_ira,
    ordered=True
)

# Converte as categorias ordenadas para códigos numéricos
dados_ira["IRA_ordinal"] = (
    dados_ira["IRA"].cat.codes + 1
)

rho_ira, p_ira = spearmanr(
    dados_ira["IRA_ordinal"],
    dados_ira["Empregabilidade"]
)

print("\n=== IRA VS. EMPREGABILIDADE ===")

print(f"Observações utilizadas: {len(dados_ira)}")
print(f"Correlação de Spearman: {rho_ira:.4f}")
print(f"Valor-p: {p_ira:.4f}")

# ============================================================
# 7. EXPORTAÇÃO DOS RESULTADOS
# ============================================================

arquivo_saida = pasta_resultados / "resultados_mercado_trabalho.xlsx"

with pd.ExcelWriter(arquivo_saida) as writer:

    dados_1.describe().to_excel(
        writer,
        sheet_name="Descritiva_Empregabilidade"
    )

    tabela_2.to_excel(
        writer,
        sheet_name="Contingencia_Teoria_Salario"
    )

    percentuais_2.to_excel(
        writer,
        sheet_name="Percentuais_Teoria_Salario"
    )

    dados_3.describe().to_excel(
        writer,
        sheet_name="Descritiva_IA"
    )

    dados_ira.groupby(
        "IRA",
        observed=False
    )["Empregabilidade"].describe().to_excel(
        writer,
        sheet_name="IRA_Empregabilidade"
    )

print("\nAnálise concluída!")

print("Resultados salvos em:")
print(arquivo_saida)