import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

# Chargement des données
# Remplacez le chemin si nécessaire :
df = pd.read_excel('analyse_telecom_consolidee_20250731_120922.xlsx')  # ou pd.read_csv('stations.csv')

# Conversion des décimales à virgule en float
if df['pourcentage_rattachees'].dtype == object:
    df['pourcentage_rattachees'] = df['pourcentage_rattachees'] \
        .str.replace(',', '.') \
        .astype(float)

# Aperçu des données
print("Aperçu des données :")
print(df.head(), "\n")

# --- Statistiques globales et par opérateur/région ---
print("Statistiques descriptives globales :")
print(df['pourcentage_rattachees'].describe(), "\n")

stats_op = df.groupby('operator')['pourcentage_rattachees']\
             .agg(['mean', 'median', 'std', 'min', 'max'])
print("Statistiques par opérateur :")
print(stats_op, "\n")

stats_reg = df.groupby('region')['pourcentage_rattachees']\
             .mean()\
             .reset_index(name='mean_pourcentage')
print("Statistiques (moyenne) par région :")
print(stats_reg.set_index('region'), "\n")

# Pivot pour heatmap
pivot = df.pivot(index='region', columns='operator', values='pourcentage_rattachees')

# --- Graphiques globaux ---
# (... comme précédemment : histogramme, boxplot, bar_opérateur, scatter, heatmap ...)  # conserver les blocs déjà écrits

# --- Graphiques et tableaux par région ---
# (… conserver la boucle de bar charts régionaux et affichage tableaux …)

# --- Carte de France des pourcentages moyens par région ---
# Lecture directe du GeoJSON en ligne (pas besoin de fichier local)
geo_url = 'https://france-geojson.gregoiredavid.fr/repo/regions.geojson'
regions_geo = gpd.read_file(geo_url)
print(f"GeoJSON chargé directement depuis le dépôt (URL)")

# Harmonisation du nom de la colonne pour la fusion
if 'nom' in regions_geo.columns:
    regions_geo = regions_geo.rename(columns={'nom':'region'})
elif 'nom_region' in regions_geo.columns:
    regions_geo = regions_geo.rename(columns={'nom_region':'region'})
elif 'region_name' in regions_geo.columns:
    regions_geo = regions_geo.rename(columns={'region_name':'region'})
# Sinon, ajustez selon le nom de champ présent

# Fusion avec les statistiques moyennes par région
mapdata = regions_geo.merge(stats_reg, on='region', how='left')

# Tracé de la carte choroplèthe
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
mapdata.plot(
    column='mean_pourcentage',
    cmap='OrRd',
    linewidth=0.5,
    edgecolor='black',
    legend=True,
    legend_kwds={'label': "% moyenne de stations rattachées", 'shrink': 0.5},
    ax=ax
)
ax.set_title("Carte choroplèthe des % moyennes de stations rattachées par région")
ax.axis('off')
plt.tight_layout()
# Enregistrement
plt.savefig('carte_france_pourcentage.png', dpi=150)
plt.close()

print("Carte sauvegardée sous : carte_france_pourcentage.png")

if __name__ == '__main__':
    plt.show()
