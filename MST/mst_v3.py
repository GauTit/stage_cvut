import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import folium
from math import radians, sin, cos, sqrt, asin

# ------------------------------
# 1. Fonctions de base
# ------------------------------

def load_stations(path: str, sep: str = ';') -> pd.DataFrame:
    """
    Chage le CSV des stations et convertit latitude/longitude en floats.
    """
    df = pd.read_csv(path, sep=sep)
    # On suppose qu'il y a une colonne 'latitude' et 'longitude' au format "xx,yy"
    df['lat'] = df['latitude'].str.replace(',', '.').astype(float)
    df['lon'] = df['longitude'].str.replace(',', '.').astype(float)
    return df

def haversine_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice NxN des distances haversine (en km) entre tous les points.
    latitudes et longitudes sont deux vecteurs 1D en degrés.
    """
    lat = np.radians(latitudes)
    lon = np.radians(longitudes)
    lat1 = lat.reshape(-1, 1)     # (N,1)
    lat2 = lat.reshape(1, -1)     # (1,N)
    lon1 = lon.reshape(-1, 1)
    lon2 = lon.reshape(1, -1)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Rayon moyen de la Terre en km

def mst_from_haversine(df: pd.DataFrame) -> nx.Graph:
    """
    À partir d'un DataFrame contenant 'lat' et 'lon', on construit la matrice de distances
    puis on calcule un MST via SciPy. On renvoie un Graph NetworkX des arêtes du MST.
    """
    coords = df[['lat','lon']].to_numpy()
    D = haversine_matrix(coords[:,0], coords[:,1])
    # On construit une matrice creuse CSR (sinon SciPy-minimum_spanning_tree ne fonctionne pas)
    sparse_D = csr_matrix(D)
    mst_sparse = minimum_spanning_tree(sparse_D)  # renvoie une matrice creuse (COO implicite)
    mst_coo = mst_sparse.tocoo()

    G = nx.Graph()
    for u, v, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        # SciPy ne double pas automatiquement (u,v) et (v,u), c'est bien une arête non orientée
        G.add_edge(int(u), int(v), weight=float(w))
    return G

# ------------------------------
# 2. Calcul de la centralité sur les arêtes du MST
# ------------------------------

def compute_edge_betweenness(mst: nx.Graph) -> dict:
    """
    Calcule la centralité d'intermédiarité (betweenness) pour chaque arête du MST.
    Renvoie un dict dont les clés sont des tuples (u,v) (avec u < v) et la valeur est le centrality-score.
    """
    # Par défaut, edge_betweenness_centrality renvoie un dict { (u,v) : score, ... }
    # avec (u,v) triés lexicographiquement. Comme c'est un Graph non orienté, (u,v) == (v,u).
    return nx.edge_betweenness_centrality(mst, normalized=True, weight='weight')

# ------------------------------
# 3. Construction de la carte Folium « filtrée »
# ------------------------------

def plot_principal_flows(
    df: pd.DataFrame,
    mst: nx.Graph,
    edge_betweenness: dict,
    quantile_threshold: float,
    output_html: str
) -> None:
    """
    Trace une carte Folium où seules les arêtes du MST dont la centralité est
    au-dessus du quantile_threshold (0..1) sont affichées en rouge épais.
    Les autres arêtes (betweenness plus faibles) sont tracées en gris clair
    ou masquées.
    """
    # 1) On récupère tous les scores de betweenness, on calcule le seuil
    scores = np.array(list(edge_betweenness.values()))
    threshold_value = np.quantile(scores, quantile_threshold)

    # 2) Initialisation de la carte
    center = [df['lat'].mean(), df['lon'].mean()]
    m = folium.Map(location=center, zoom_start=8)

    # 3) On trace d'abord toutes les stations en points (noirs)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=1,
            color='black',
            fill=True,
            fill_opacity=1
        ).add_to(m)

    # 4) On boucle sur les arêtes du MST et on colore selon qu'on dépasse le seuil ou pas
    for (u, v), score in edge_betweenness.items():
        p1 = (df.at[u, 'lat'], df.at[u, 'lon'])
        p2 = (df.at[v, 'lat'], df.at[v, 'lon'])
        if score >= threshold_value:
            # Arête « principale » : on met en rouge épais, opacité forte
            folium.PolyLine(
                [p1, p2],
                color='red',
                weight=4,
                opacity=0.9
            ).add_to(m)
        else:
            # Arête « secondaire » : on trace en gris clair / très fin / opacité faible
            folium.PolyLine(
                [p1, p2],
                color='#cccccc',
                weight=1,
                opacity=0.3
            ).add_to(m)

    # 5) Sauvegarde
    m.save(output_html)
    print(f"Carte avec flux principaux sauvegardée dans : {output_html}")

# ------------------------------
# 4. Exemple d'utilisation
# ------------------------------
if __name__ == "__main__":
    # 4.1. Chargement
    df = load_stations('stations_normandie_orange.csv', sep=';')

    # 4.2. Construction du MST (NetworkX)
    mst_graph = mst_from_haversine(df)

    # 4.3. Calcul de la centralité des arêtes
    eb = compute_edge_betweenness(mst_graph)

    # 4.4. Choix d'un quantile : par exemple, on ne veut garder QUE les 20% d'arêtes les plus "centrales"
    quantile = 0.80  # 80e percentile → on affiche en rouge les arêtes dont betweenness ≥ quantile_80
    plot_principal_flows(df, mst_graph, eb, quantile_threshold=quantile,
                          output_html='mst_flux_principaux.html')
