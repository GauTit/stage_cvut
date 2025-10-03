import pandas as pd
import numpy as np
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.neighbors import BallTree

import folium

# ---------------------------------------
# 1. Chargement des stations
# ---------------------------------------
def load_stations(path: str, sep: str = ';') -> pd.DataFrame:
    """
    Charge le CSV des stations, convertit latitude/longitude (format "xx,yy") en floats (°).
    """
    df = pd.read_csv(path, sep=sep)
    # On suppose qu'il y a des colonnes 'latitude' et 'longitude' au format "xx,yy"
    df['lat'] = df['latitude'].str.replace(',', '.').astype(float)
    df['lon'] = df['longitude'].str.replace(',', '.').astype(float)
    return df

# ---------------------------------------
# 2. Construction du graphe k-NN via BallTree
# ---------------------------------------
def build_sparse_knn_graph(
    df: pd.DataFrame,
    k_neighbors: int = 5
) -> csr_matrix:
    """
    À partir d'un DataFrame contenant 'lat' et 'lon' en degrés,
    construit un graphe creux CSR contenant, pour chaque station,
    les k plus proches voisins (poids = distance en km).
    On utilise sklearn.neighbors.BallTree avec la métrique haversine.
    """
    # 1) Conversion des lat/lon en radians, comme l'exige BallTree (metric='haversine').
    coords_rad = np.vstack([
        np.radians(df['lat'].values),
        np.radians(df['lon'].values)
    ]).T  # shape (N, 2)

    # 2) Construction du BallTree (métrique haversine)
    #    => les distances renvoyées seront en radians d'angle.
    tree = BallTree(coords_rad, metric='haversine')

    # 3) Pour chaque point, on récupère les k+1 plus proches (y compris lui-même)
    #    distances en radians, indices des voisins
    #    (k_neighbors+1 car le plus proche de i est i-même, à distance 0)
    dists_rad, indices = tree.query(coords_rad, k=k_neighbors + 1)

    # 4) Conversion distance angulaire → distance en km
    earth_radius = 6371.0
    dists_km = dists_rad * earth_radius  # shape (N, k+1)

    N = df.shape[0]
    # On prépare les listes pour construire la matrice creuse
    row_idx = []
    col_idx = []
    data_wt = []

    for i in range(N):
        for j in range(1, k_neighbors + 1):  # on saute j=0 (c'est la station elle-même)
            neighbor = indices[i, j]
            dist_ij = dists_km[i, j]

            # On ajoute arête (i, neighbor)
            row_idx.append(i)
            col_idx.append(neighbor)
            data_wt.append(dist_ij)

            # Comme on veut un graphe non orienté, on ajoute aussi (neighbor, i)
            row_idx.append(neighbor)
            col_idx.append(i)
            data_wt.append(dist_ij)

    # Construction du sparse matrix (format CSR)
    sparse_mat = csr_matrix(
        (data_wt, (row_idx, col_idx)),
        shape=(N, N)
    )

    return sparse_mat

# ---------------------------------------
# 3. Calcul du MST sur le graphe creux
# ---------------------------------------
def mst_from_sparse_graph(sparse_graph: csr_matrix) -> nx.Graph:
    """
    Applique l'algorithme minimum_spanning_tree de SciPy sur une matrice CSR creuse.
    Retourne un Graph NetworkX avec les arêtes du MST (poids = weight).
    """
    mst_sparse = minimum_spanning_tree(sparse_graph)  # renvoie un scipy.sparse.csr_matrix
    mst_coo = mst_sparse.tocoo()

    G = nx.Graph()
    for u, v, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        # SciPy renvoie seulement u→v pour un graphe non orienté, on ajoute donc une seule arête
        G.add_edge(int(u), int(v), weight=float(w))
    return G

# ---------------------------------------
# 4. Centralité d'intermédiarité des arêtes
# ---------------------------------------
def compute_edge_betweenness(mst: nx.Graph) -> dict:
    """
    Calcule la centralité d'intermédiarité (betweenness) pour chaque arête du MST.
    Renvoie un dict { (u,v) : score }, avec u < v par convention.
    """
    # Note : edge_betweenness_centrality retourne par défaut { (u,v): score, ... }
    # Avec (u,v) triés lexicographiquement, et comme c'est un Graph non orienté, (u,v) == (v,u).
    return nx.edge_betweenness_centrality(mst, normalized=True, weight='weight')

# ---------------------------------------
# 5. Tracé sur Folium des flux principaux
# ---------------------------------------
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
    Les autres arêtes (betweenness plus faibles) sont tracées en gris clair/maigres.
    """
    # 1) Extraction des scores
    scores = np.array(list(edge_betweenness.values()))
    threshold_value = np.quantile(scores, quantile_threshold)

    # 2) Initialisation de la carte (centrée sur la moyenne des points)
    center = [df['lat'].mean(), df['lon'].mean()]
    m = folium.Map(location=center, zoom_start=8)

    # 3) On trace d'abord toutes les stations en tant que cercles noirs
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=1.5,
            color='black',
            fill=True,
            fill_opacity=1
        ).add_to(m)

    # 4) On boucle sur les arêtes du MST et on colore selon le seuil
    for (u, v), score in edge_betweenness.items():
        p1 = (df.at[u, 'lat'], df.at[u, 'lon'])
        p2 = (df.at[v, 'lat'], df.at[v, 'lon'])
        if score >= threshold_value:
            # Flux « principal » : rouge épais, opacité forte
            folium.PolyLine(
                [p1, p2],
                color='red',
                weight=4,
                opacity=0.9
            ).add_to(m)
        else:
            # Flux « secondaire » : gris clair, trait fin, opacité faible
            folium.PolyLine(
                [p1, p2],
                color='#cccccc',
                weight=1,
                opacity=0.3
            ).add_to(m)

    # 5) Sauvegarde du fichier HTML
    m.save(output_html)
    print(f"Carte avec flux principaux enregistrée dans : {output_html}")

# ---------------------------------------
# 6. Exemple d'utilisation
# ---------------------------------------
if __name__ == "__main__":
    # 6.1. Chargement du DataFrame
    df = load_stations('stations_normandie_orange.csv', sep=';')

    # 6.2. Préparation du graphe creux via k-NN (ic = 10 voisins)
    k_neighbors = 10
    sparse_graph = build_sparse_knn_graph(df, k_neighbors=k_neighbors)

    # 6.3. Calcul du MST à partir du graphe creux
    mst_graph = mst_from_sparse_graph(sparse_graph)

    # 6.4. Centralité d'intermédiarité des arêtes du MST
    eb = compute_edge_betweenness(mst_graph)

    # 6.5. Choix du quantile (ex. on ne garde que les 20 % d’arêtes les plus centrales)
    quantile_value = 0.80  # 80ᵉ percentile

    # 6.6. Tracé de la carte Folium
    output_file = 'mst_flux_principaux_kNN.html'
    plot_principal_flows(
        df,
        mst_graph,
        eb,
        quantile_threshold=quantile_value,
        output_html=output_file
    )
