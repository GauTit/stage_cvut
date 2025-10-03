import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Charger le CSV filtré
df = pd.read_csv('stations_normandie_orange.csv', sep=';')

# Conversion des coordonnées en float
df['latitude'] = df['latitude'].str.replace(',', '.').astype(float)
df['longitude'] = df['longitude'].str.replace(',', '.').astype(float)

# Préparation des données pour DBSCAN géographique (radians)
coords = np.radians(df[['latitude', 'longitude']].values)

# Paramètres DBSCAN : 1 km de rayon, 5 points minimum
kms_per_radian = 6371.0088
epsilon = 6.0 / kms_per_radian

db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine')
df['cluster'] = db.fit_predict(coords)

# Création de la carte centrée sur la Normandie
center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=center, zoom_start=8)

# Attribution de couleurs pour chaque cluster
unique_clusters = sorted(df['cluster'].unique())
colormap = cm.get_cmap('tab20', len(unique_clusters))

for idx, cluster in enumerate(unique_clusters):
    hex_color = mcolors.to_hex(colormap(idx))
    subset = df[df['cluster'] == cluster]
    for _, row in subset.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            popup=f"Cluster {cluster}"
        ).add_to(m)

# Sauvegarde de la carte
output_html = './density_scans/dbscan_clusters_normandie_orange_6km.html'
m.save(output_html)


