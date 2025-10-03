import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import folium
from math import radians, sin, cos, sqrt, asin

# 1. Chargement et préparation des données
df = pd.read_csv('stations_normandie_orange.csv', sep=';')
df['lat'] = df['latitude'].str.replace(',', '.').astype(float)
df['lon'] = df['longitude'].str.replace(',', '.').astype(float)
coords = df[['lat','lon']].values

# 2. Fonction haversine pour calculer la distance en kilomètres
def haversine_pair(u, v):
    lat1, lon1 = radians(u[0]), radians(u[1])
    lat2, lon2 = radians(v[0]), radians(v[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

# 3. Calcul de la matrice de distances
#    On utilise pdist + squareform pour efficacité
dists = pdist(coords, lambda u, v: haversine_pair(u, v))
D = squareform(dists)  # matrice symétrique

# 4. Construction du graphe complet et extraction du MST
G = nx.Graph()
n = len(df)
for i in range(n):
    for j in range(i+1, n):
        G.add_edge(i, j, weight=D[i, j])
T = nx.minimum_spanning_tree(G)

# 5. Création de la carte Folium
center = [df['lat'].mean(), df['lon'].mean()]
m = folium.Map(location=center, zoom_start=8)

# a) Tracer les arêtes du MST en rouge
for u, v in T.edges():
    p1 = (df.loc[u, 'lat'], df.loc[u, 'lon'])
    p2 = (df.loc[v, 'lat'], df.loc[v, 'lon'])
    folium.PolyLine([p1, p2], color='red', weight=2, opacity=0.7).add_to(m)

# b) Ajouter les stations en cercles noirs
for _, row in df.iterrows():
    folium.CircleMarker(
        location=(row['lat'], row['lon']),
        radius=2,
        color='black',
        fill=True,
        fill_opacity=1
    ).add_to(m)

# 6. Sauvegarde de la carte
output_html = 'mst_stations_normandie_orange.html'
m.save(output_html)

print(f"MST calculé et carte sauvegardée dans : {output_html}")
