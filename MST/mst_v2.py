import pandas as pd
import numpy as np
import networkx as nx
import folium
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict
from tqdm import tqdm

# ------------------------------
# 1. Fonctions de base optimisées
# ------------------------------

def load_stations(path: str, sep: str = ';') -> pd.DataFrame:
    """Charge et nettoie les données des stations"""
    df = pd.read_csv(path, sep=sep)
    
    # Normalisation des colonnes
    col_mapping = {
        'latitude': 'lat',
        'longitude': 'lon',
        'Latitude': 'lat',
        'Longitude': 'lon'
    }
    df = df.rename(columns=col_mapping)
    
    # Conversion des coordonnées
    for col in ['lat', 'lon']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.').astype(float)
    
    return df.dropna(subset=['lat', 'lon']).reset_index(drop=True)

def haversine(lat1, lon1, lat2, lon2):
    """Calcule la distance haversine entre deux points"""
    R = 6371  # Rayon terrestre en km
    
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    return R * c

# ------------------------------
# 2. Clustering spatial et triangulation
# ------------------------------

def spatial_clustering(df: pd.DataFrame, eps_km: float = 2, min_samples: int = 2) -> pd.DataFrame:
    """Clusterise les stations spatialement avec DBSCAN"""
    coords = df[['lat', 'lon']].values
    
    # Conversion eps en degrés approximative
    eps_deg = eps_km / 111
    
    # Clustering
    db = DBSCAN(eps=eps_deg, min_samples=min_samples, metric='haversine')
    labels = db.fit_predict(np.radians(coords))
    
    df['cluster'] = labels
    return df

def compute_intercluster_links(df: pd.DataFrame, max_links: int = 3) -> list:
    """Identifie les liens principaux entre clusters via triangulation de Delaunay"""
    # Calcul des centroïdes de clusters
    centroids = df.groupby('cluster')[['lat', 'lon']].mean().reset_index()
    centroids = centroids[centroids['cluster'] != -1]  # Exclure le bruit
    
    if len(centroids) < 3:
        return []
    
    # Triangulation de Delaunay
    points = centroids[['lat', 'lon']].values
    tri = Delaunay(points)
    
    # Collecte des liens entre clusters
    cluster_links = defaultdict(list)
    
    # Parcourir les simplexes de la triangulation
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                c1 = centroids.iloc[simplex[i]]['cluster']
                c2 = centroids.iloc[simplex[j]]['cluster']
                
                # Calcul de la distance
                p1 = points[simplex[i]]
                p2 = points[simplex[j]]
                distance = haversine(p1[0], p1[1], p2[0], p2[1])
                
                cluster_links[(min(c1, c2), (max(c1, c2)))].append(distance)
    
    # Sélection des liens principaux
    principal_links = []
    for (c1, c2), distances in cluster_links.items():
        # Prendre la distance moyenne
        avg_distance = np.mean(distances)
        principal_links.append((c1, c2, avg_distance))
    
    # Trier par distance et limiter le nombre de liens
    principal_links.sort(key=lambda x: x[2])
    return principal_links[:max_links]

# ------------------------------
# 3. Identification des axes structurants
# ------------------------------

def find_main_axes(df: pd.DataFrame, principal_links: list) -> list:
    """Identifie les axes principaux entre clusters"""
    main_axes = []
    
    for c1, c2, _ in principal_links:
        # Stations des clusters connectés
        cluster1 = df[df['cluster'] == c1]
        cluster2 = df[df['cluster'] == c2]
        
        # Trouver les paires de stations les plus proches entre clusters
        min_distance = float('inf')
        best_pair = None
        
        for _, row1 in cluster1.iterrows():
            for _, row2 in cluster2.iterrows():
                distance = haversine(row1['lat'], row1['lon'], row2['lat'], row2['lon'])
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (row1.name, row2.name)
        
        if best_pair:
            main_axes.append(best_pair)
    
    return main_axes

# ------------------------------
# 4. Visualisation améliorée
# ------------------------------

def plot_cluster_flows(
    df: pd.DataFrame,
    main_axes: list,
    output_html: str,
    include_all_links: bool = False
) -> None:
    """Crée une visualisation des flux principaux entre clusters"""
    # Centre de la carte
    center = [df['lat'].mean(), df['lon'].mean()]
    m = folium.Map(location=center, zoom_start=9, tiles='CartoDB positron')
    
    # Palette de couleurs pour les clusters
    cluster_colors = {
        -1: '#cccccc',  # Bruit
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
        4: '#9467bd',
        5: '#8c564b',
        6: '#e377c2',
        7: '#7f7f7f',
        8: '#bcbd22',
        9: '#17becf'
    }
    
    # Ajouter les stations avec couleur par cluster
    for _, row in df.iterrows():
        color = cluster_colors.get(row['cluster'], '#000000')
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=f"Cluster: {row['cluster']}"
        ).add_to(m)
    
    # Ajouter les liens principaux
    for station1, station2 in main_axes:
        p1 = (df.at[station1, 'lat'], df.at[station1, 'lon'])
        p2 = (df.at[station2, 'lat'], df.at[station2, 'lon'])
        
        folium.PolyLine(
            [p1, p2],
            color='red',
            weight=4,
            opacity=0.9,
            dash_array='5, 10'
        ).add_to(m)
    
    # Optionnel: ajouter tous les liens intra-cluster
    if include_all_links:
        for cluster_id in df['cluster'].unique():
            if cluster_id == -1:
                continue
                
            cluster_points = df[df['cluster'] == cluster_id]
            points = cluster_points[['lat', 'lon']].values
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    folium.PolyLine(
                        [points[i], points[j]],
                        color='#aaaaaa',
                        weight=1,
                        opacity=0.2
                    ).add_to(m)
    
    # Légende
    legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; 
                    background: white; padding: 10px; border: 1px solid grey; z-index: 1000;">
            <h4>Légende</h4>
            <p><span style="color: red; font-weight: bold">━━━</span> Axes principaux</p>
            <p><span style="color: #1f77b4">⏺</span> Stations clusterisées</p>
            <p><span style="color: #cccccc">⏺</span> Stations isolées</p>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(output_html)
    print(f"Carte générée: {output_html}")

# ------------------------------
# 5. Exemple d'utilisation
# ------------------------------
if __name__ == "__main__":
    # Chargement des données
    df = load_stations('stations_normandie_orange.csv', sep=';')
    
    # Clustering spatial
    print("Clustering spatial...")
    df = spatial_clustering(df, eps_km=1, min_samples=3)
    
    # Identification des liens entre clusters
    print("Calcul des liens inter-clusters...")
    principal_links = compute_intercluster_links(df, max_links=5)
    
    # Identification des axes principaux
    print("Identification des axes structurants...")
    main_axes = find_main_axes(df, principal_links)
    
    # Visualisation
    print("Création de la carte...")
    plot_cluster_flows(
        df=df,
        main_axes=main_axes,
        output_html='flux_principaux_clusters.html',
        include_all_links=False
    )