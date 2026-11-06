# -*- coding: utf-8 -*-
"""
Classificação de Perfis e Geração de Gráficos da Roda da Vida
Processamento de respostas do Google Forms com escala Likert (1 a 5)
convertida para a métrica clássica (1 a 10).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

file_path = Path("Pesquisa- Atividade-Dia 2 (Responses).xlsx")

print("Diretório atual:", Path.cwd())
print("Arquivo encontrado:", file_path.exists())

df = pd.read_excel(file_path)

# -------------------------------------------------------------
# 2. SELEÇÃO E TRATAMENTO DAS DIMENSÕES DA RODA DA VIDA
# -------------------------------------------------------------
# Identifica as colunas da matriz da roda da vida (avaliação atual)
current_cols = [c for c in df.columns if 'atualmente' in c and '[' in c]

# Nomes limpos de cada uma das 12 dimensões
dim_names = [c.split('[')[1].replace(']', '').strip() for c in current_cols]

# Rótulos para os vértices do gráfico radar (com quebra de linha visual)
labels_radar = [
    'Saúde',
    'Desenvolvimento\nIntelectual',
    'Equilíbrio\nEmocional',
    'Propósito',
    'Recursos\nFinanceiros',
    'Contribuição\nSocial',
    'Família',
    'Relacionamento\nAmoroso',
    'Vida Social',
    'Hobbies',
    'Plenitude',
    'Espiritualidade',
]

# -------------------------------------------------------------
# 3. TRANSFORMAÇÃO E NORMALIZAÇÃO DE ESCALAS (1–5 para 1–10)
# -------------------------------------------------------------
# Matriz bruta de notas (1 a 5)
X_raw = df[current_cols].values

# Reescalonamento linear Min-Max:
# Nota_10 = 1 + ((X - 1) / (5 - 1)) * 9
X_scaled = 1.0 + (X_raw - 1.0) * (9.0 / 4.0)

# DataFrame padronizado com as notas de 1 a 10
df_scaled = pd.DataFrame(X_scaled, columns=dim_names)


# -------------------------------------------------------------
# 4. FUNÇÃO DE CLASSIFICAÇÃO DOS 5 PERFIS
# -------------------------------------------------------------
def classificar_perfil(row):
  """Classifica cada indivíduo com base nas métricas descritivas do vetor:

  row[0]: Saúde e disposição          row[6]: Família
  row[1]: Desenvolvimento intelectual  row[7]: Relacionamento amoroso
  row[2]: Equilíbrio emocional         row[8]: Vida Social
  row[3]: Realização e propósito       row[9]: Hobbies e diversão
  row[4]: Recursos financeiros         row[10]: Plenitude e felicidade
  row[5]: Contribuição social          row[11]: Espiritualidade
  """
  mean_val = np.mean(row)
  std_val = np.std(row)
  min_val = np.min(row)
  max_val = np.max(row)
  range_val = max_val - min_val

  # Médias de Macrodimensões
  prof_fin = np.mean([row[1], row[3], row[4]])  # Carreira e Finanças
  afetivo = np.mean(
      [row[6], row[7], row[10], row[11]]
  )  # Afeto, Família e Espiritualidade
  saude_lazer = np.mean([row[0], row[8], row[9]])  # Vitalidade, Social e Lazer

  pct_abaixo_5 = np.mean(row < 5.0)

  # 5. Perfil Sobrevivente: Média baixa (< 4.5) e maioria das notas (< 5.0)
  if mean_val < 4.5 and pct_abaixo_5 >= 0.6:
    return 'Sobrevivente'

  # 1. Perfil Roda Perfeita: Notas uniformes e altas (baixa dispersão)
  if mean_val >= 7.0 and std_val <= 1.8 and min_val >= 5.0:
    return 'Roda Perfeita'

  # 2. Perfil Workaholic: Foco em Carreira/Finanças (>= 7.0) e queda em Afeto/Saúde (<= 4.5)
  if prof_fin >= 7.0 and (saude_lazer <= 4.5 or afetivo <= 4.5):
    return 'Workaholic'

  # 3. Perfil Família e Lar: Foco no Afeto/Família (>= 7.0) com defasagem em Carreira/Finanças (<= 5.5)
  if afetivo >= 7.0 and prof_fin <= 5.5:
    return 'Família e Lar'

  # 4. Perfil Montanha-Russa: Grande variabilidade (amplitude >= 6.0 e desvio >= 2.2)
  if range_val >= 6.0 and std_val >= 2.2:
    return 'Montanha-Russa'

  # Critérios de proximidade para casos limítrofes
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


# Aplica nos dados
df['Perfil'] = [classificar_perfil(r) for r in X_scaled]
df_scaled['Perfil'] = df['Perfil']

# Calcula os centroides (médias) de cada dimensão por perfil
profile_means = df_scaled.groupby('Perfil')[dim_names].mean()

# -------------------------------------------------------------
# 5. GERAÇÃO DOS GRÁFICOS RADAR (Painel 2x3)
# -------------------------------------------------------------
profile_order = [
    'Roda Perfeita',
    'Workaholic',
    'Família e Lar',
    'Montanha-Russa',
    'Sobrevivente',
]
colors = ['#2ecc71', '#e67e22', '#3498db', '#9b59b6', '#e74c3c']

num_vars = len(labels_radar)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Fecha o círculo polar

fig, axes = plt.subplots(
    2, 3,
    figsize=(20, 15),
    subplot_kw={"polar": True}
)

axes = axes.flatten()


# Renderização individual dos 5 perfis
for idx, (p_name, color) in enumerate(zip(profile_order, colors)):
  ax = axes[idx]
  values = profile_means.loc[p_name].values.tolist()
  values += values[:1]

  ax.plot(angles, values, color=color, linewidth=2.5, linestyle='solid')
  ax.fill(angles, values, color=color, alpha=0.25)

  ax.set_theta_offset(np.pi / 2)
  ax.set_theta_direction(-1)
  ax.set_thetagrids(np.degrees(angles[:-1]), labels_radar, fontsize=8.5)

  ax.set_ylim(0, 10)
  ax.set_yticks([2, 4, 6, 8, 10])
  ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=7.5, color='gray')

  n_count = (df['Perfil'] == p_name).sum()
  pct = (n_count / len(df)) * 100
  ax.set_title(
    f"{p_name}\n"
    f"({n_count} respondentes | {pct:.1f}%)",
    fontsize=11,
    color=color,
    fontweight="bold",
    pad=25
)

# 6º Gráfico: Comparativo Geral Sobreposto
ax_comp = axes[5]

for p_name, color in zip(profile_order, colors):
    values = profile_means.loc[p_name].values.tolist()
    values += values[:1]

    ax_comp.plot(
        angles,
        values,
        color=color,
        linewidth=1.8,
        label=p_name
    )

ax_comp.set_theta_offset(np.pi / 2)
ax_comp.set_theta_direction(-1)

ax_comp.set_thetagrids(
    np.degrees(angles[:-1]),
    labels_radar,
    fontsize=8
)

ax_comp.set_ylim(0, 10)
ax_comp.set_yticks([2, 4, 6, 8, 10])
ax_comp.set_yticklabels(
    ["2", "4", "6", "8", "10"],
    fontsize=7,
    color="gray"
)

ax_comp.set_title(
    "Comparativo geral dos perfis",
    fontsize=11,
    color="#2c3e50",
    fontweight="bold",
    pad=25
)

ax_comp.legend(
    loc="upper left",
    bbox_to_anchor=(1.15, 1.05),
    fontsize=8,
    frameon=True
)
fig.subplots_adjust(
    left=0.05,
    right=0.88,
    top=0.93,
    bottom=0.07,
    wspace=0.35,
    hspace=0.45
)

plt.savefig(
    "radar_perfis_roda_da_vida.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()