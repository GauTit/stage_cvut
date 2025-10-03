import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Chargement des données
# Remplacez le chemin si nécessaire :
df = pd.read_excel('analyse_telecom_consolidee_20250731_120922.xlsx')  # ou pd.read_csv('stations.csv')

# Conversion des décimales à virgule en float
if df['pourcentage_rattachees'].dtype == object:
    df['pourcentage_rattachees'] = df['pourcentage_rattachees'] \
        .str.replace(',', '.') \
        .astype(float)

# Affichage des  premières lignes pour vérification
print("Aperçu des données :")
print(df.head(), "\n")

# Statistiques descriptives globales
print("Statistiques descriptives globales :")
print(df.describe(), "\n")

# Statistiques par opérateur
stats_op = df.groupby('operator')['pourcentage_rattachees'] \
             .agg(['mean', 'median', 'std', 'min', 'max'])
print("\nStatistiques par opérateur :")
print(stats_op, "\n")

# Statistiques par région
stats_reg = df.groupby('region')['pourcentage_rattachees'] \
             .agg(['mean', 'median', 'std', 'min', 'max'])
print("Statistiques par région :")
print(stats_reg, "\n")

# Pivot pour heatmap
pivot = df.pivot(index='region', columns='operator', values='pourcentage_rattachees')

# --- Graphiques ---

# 1. Histogramme de la distribution des % rattachées
plt.figure()
df['pourcentage_rattachees'].hist()
plt.title('Distribution des pourcentages de stations rattachées')
plt.xlabel('% rattachées')
plt.ylabel('Nombre de régions/opérateurs')
plt.tight_layout()
plt.savefig('hist_distribution.png')

# 2. Boxplot des % rattachées par opérateur
plt.figure()
df.boxplot(column='pourcentage_rattachees', by='operator')
plt.title('Boxplot des % rattachées par opérateur')
plt.suptitle('')  # supprime le titre automatique
plt.xlabel('Opérateur')
plt.ylabel('% rattachées')
plt.tight_layout()
plt.savefig('boxplot_operator.png')

# 3. Bar chart : moyenne de % rattachées par opérateur
plt.figure()
stats_op['mean'].plot(kind='bar')
plt.title("Moyenne du pourcentage de stations rattachées par opérateur")
plt.xlabel('Opérateur')
plt.ylabel('Moyenne % rattachées')
plt.tight_layout()
plt.savefig('bar_mean_operator.png')

# 4. Scatter plot : Total vs Stations rattachées
plt.figure()
plt.scatter(df['total_stations'], df['stations_rattachees'])
plt.title('Total des stations vs stations rattachées')
plt.xlabel('Total stations')
plt.ylabel('Stations rattachées')
# Ligne de régression simple
a, b = np.polyfit(df['total_stations'], df['stations_rattachees'], 1)
plt.plot(df['total_stations'], a*df['total_stations'] + b)
plt.tight_layout()
plt.savefig('scatter_total_vs_rattachees.png')

# 5. Heatmap des pourcentages par région et opérateur
plt.figure()
plt.imshow(pivot, aspect='auto', interpolation='none')
plt.colorbar(label='% rattachées')
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title('Heatmap des % rattachées (région x opérateur)')
plt.tight_layout()
plt.savefig('heatmap_region_operator.png')

print("\nGraphiques enregistrés sous :")
print("  - hist_distribution.png")
print("  - boxplot_operator.png")
print("  - bar_mean_operator.png")
print("  - scatter_total_vs_rattachees.png")
print("  - heatmap_region_operator.png")

if __name__ == '__main__':
    plt.show()
