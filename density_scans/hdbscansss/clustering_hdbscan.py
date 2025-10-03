import pandas as pd
import numpy as np
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import hdbscan

# 1. Chargement du CSV filtré
df = pd.read_csv('stations_normandie_orange.csv', sep=';')

# 2. Conversion des coordonnées en float
df['latitude']  = df['latitude'].str.replace(',', '.').astype(float)
df['longitude'] = df['longitude'].str.replace(',', '.').astype(float)

# 3. Préparation des données pour HDBSCAN (on peut travailler directement en degrés)
coords = df[['latitude', 'longitude']].values

# 4. Exécution du clustering HDBSCAN
#    min_cluster_size = taille minimale d’un cluster (ajustez selon vos données)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,
    metric='haversine',       # utiliser haversine pour distances géographiques
    cluster_selection_method='eom'
)
# Si metric='haversine', il faut convertir en radians :
coords_rad = np.radians(coords)
df['cluster'] = clusterer.fit_predict(coords_rad)

# 5. Création de la carte Folium
center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=center, zoom_start=8)

# Palette de couleurs
clusters = sorted(df['cluster'].unique())
cmap = cm.get_cmap('tab20', len(clusters))

for idx, cl in enumerate(clusters):
    color = mcolors.to_hex(cmap(idx))
    subset = df[df['cluster'] == cl]
    for _, row in subset.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"Cluster {cl}"
        ).add_to(m)

# 6. Sauvegarde de la carte
output_html = './density_scans/hdbscansss/hdbscan_clusters_normandie_orange.html'
m.save(output_html)
print(f"Carte HDBSCAN enregistrée dans : {output_html}")
