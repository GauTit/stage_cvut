import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, box, LineString
from shapely.ops import nearest_points
import pandas as pd
import numpy as np
import folium
from matplotlib import colormaps
import matplotlib.colors as mcolors
import openpyxl
import math

# Configuration OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True

# Charger les données de stations
stations_df = pd.read_csv(
    "stations_normandie_orange.csv",
    sep=None,
    engine='python',
    decimal=','
)
# Détecter le nom de la station
name_col = next(
    (col for col in stations_df.columns if col.lower() not in ['latitude', 'longitude', 'lat', 'lon', 'lng']),
    stations_df.columns[0]
)

# Créer GeoDataFrame des stations
stations_gdf = gpd.GeoDataFrame(
    stations_df,
    geometry=stations_df.apply(lambda r: Point(r.longitude, r.latitude), axis=1),
    crs="EPSG:4326"
)

# Projection en Lambert-93 (mètres)
stations_proj = stations_gdf.to_crs("EPSG:2154")

# Définir zone d'intérêt (buffer 10 km)
minx, miny, maxx, maxy = stations_proj.total_bounds
buffer = 10000  # mètres
roi = box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

# 1. Télécharger le réseau routier (motorway à primary_link)
road_filter = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link"]'
G_road = ox.graph_from_place(
    "Normandie, France",
    network_type=None,
    custom_filter=road_filter,
    retain_all=True
)
nodes_road, edges_road = ox.graph_to_gdfs(G_road)
edges_road = edges_road.to_crs("EPSG:2154")
edges_road = edges_road[edges_road.intersects(roi)].copy()

# 2. Télécharger les voies ferrées principales
rail_filter = '["railway"="rail"]'
G_rail = ox.graph_from_place(
    "Normandie, France",
    network_type=None,
    custom_filter=rail_filter,
    retain_all=True
)
nodes_rail, edges_rail = ox.graph_to_gdfs(G_rail)
edges_rail = edges_rail.to_crs("EPSG:2154")
edges_rail = edges_rail[edges_rail.intersects(roi)].copy()

# Sous-ensembles des géométries OSM avec attributs utiles
def subset_osm(edges):
    cols = ['geometry', 'osmid']
    for extra in ['ref', 'name']:
        if extra in edges.columns:
            cols.append(extra)
    return edges[cols].copy()

edges_comb = pd.concat([subset_osm(edges_road), subset_osm(edges_rail)], ignore_index=True)

# Spatial join pour le tronçon le plus proche à moins de 1 km
stations_joined = gpd.sjoin_nearest(
    stations_proj,
    edges_comb,
    how='left',
    distance_col='dist_to_highway_m',
    max_distance=1000
)

# Vérifier si on a des stations valides
if stations_joined.empty or stations_joined['dist_to_highway_m'].isna().all():
    print("ATTENTION: Aucune station trouvée à moins de 1 km d'une route principale!")
    print("Vous devriez vérifier:")
    print("1. Le fichier CSV des stations")
    print("2. Les coordonnées des stations")
    print("3. Peut-être augmenter la distance de recherche")
    exit()

# Renommer et nettoyer
stations_all_routes = (
    stations_joined
    .rename(columns={'osmid': 'nearest_osmid'})
    .drop(columns=['index_right'])
)

# Gestion des osmids multiples (liste -> tuple)
def create_osmid_key(osmid):
    if isinstance(osmid, list):
        return tuple(sorted(osmid))
    if pd.isna(osmid):
        return None
    return (osmid,)

stations_all_routes['osmid_key'] = stations_all_routes['nearest_osmid'].apply(create_osmid_key)

# Assurer la présence des colonnes 'ref' et 'name'
for col in ['ref', 'name']:
    if col not in stations_all_routes.columns:
        stations_all_routes[col] = np.nan

# Création de la clé parent (ref > name > osmid)
def create_parent_key(row):
    ref = row['ref']
    if not (isinstance(ref, float) and math.isnan(ref)):
        if isinstance(ref, list) and ref:
            return f"ref:{';'.join(map(str, ref))}"
        if isinstance(ref, str) and ref.strip():
            return f"ref:{ref.strip()}"
        if not isinstance(ref, (list, str)):
            return f"ref:{ref}"
    name = row['name']
    if isinstance(name, str) and name.strip():
        return f"name:{name.strip()}"
    return create_osmid_key(row['nearest_osmid'])

stations_all_routes['parent_key'] = stations_all_routes.apply(create_parent_key, axis=1)

# Utiliser stations_all_routes pour le reste du traitement
stations_proj = stations_all_routes.copy()

# Statistiques par route et export Excel
route_stats = (
    stations_proj.groupby('parent_key')
    .size()
    .reset_index(name='station_count')
    .sort_values('station_count', ascending=False)
)

# Vérification et affichage des statistiques
if not route_stats.empty:
    top = route_stats.iloc[0]
    print(f"Route avec le plus de stations : {top['parent_key']} ({top['station_count']} stations)")
    print(f"Nombre total de routes trouvées : {len(route_stats)}")
    print(f"Nombre total de stations liées : {stations_proj.shape[0]}")
else:
    print("Aucune route trouvée avec des stations associées")

stations_export = stations_proj.copy()
# Correction: utiliser les coordonnées de la géométrie correctement
stations_export['latitude'] = stations_export.geometry.to_crs("EPSG:4326").y
stations_export['longitude'] = stations_export.geometry.to_crs("EPSG:4326").x
stations_export = stations_export.drop(columns=['geometry'])

output_excel = "stations_et_routes_summary.xlsx"
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    stations_export[
        ['latitude','longitude','nearest_osmid','ref','name','dist_to_highway_m','parent_key']
    ].to_excel(writer, sheet_name='Stations', index=False)
    route_stats.to_excel(writer, sheet_name='Route_Stats', index=False)

print(f"Fichier Excel généré : {output_excel}")

# Calcul des liens entre voisins sur la même route avec TSP OUVERT (mètres en EPSG:2154)
# Filtrer uniquement les stations valides (< 1km d'une route)
valid = stations_proj.dropna(subset=['dist_to_highway_m'])
valid = valid[valid['dist_to_highway_m'] < 1000]
neighbor_geoms = []

def solve_tsp_open(points):
    """Résoudre TSP OUVERT avec algorithme glouton (nearest neighbor) - pas de retour au départ"""
    if len(points) <= 2:
        return list(range(len(points)))
    
    # Matrice des distances
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = points[i]['geometry'].distance(points[j]['geometry'])
    
    # Algorithme du plus proche voisin SANS retour au début
    unvisited = set(range(1, n))
    current = 0
    tour = [current]
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: distances[current][x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    # PAS de retour au point de départ (TSP ouvert)
    return tour

# Vérifier s'il y a des stations valides avant de continuer
if valid.empty:
    print("Aucune station valide trouvée pour créer les liens entre voisins")
    neighbors_wgs = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
else:
    for pk, group in valid.groupby('parent_key'):
        if len(group) < 2:
            continue
        
        # Créer une liste des stations uniques pour cette route
        stations_dict = {}
        for idx, row in group.iterrows():
            station_id = (row.geometry.x, row.geometry.y)  # Utiliser les coordonnées comme clé unique
            if station_id not in stations_dict:
                stations_dict[station_id] = {
                    'idx': idx,
                    'geometry': row.geometry,
                    'coords': (row.geometry.x, row.geometry.y)
                }
        
        stations_list = list(stations_dict.values())
        
        # Cas simple : 2 stations seulement
        if len(stations_list) == 2:
            neighbor_geoms.append(LineString([stations_list[0]['coords'], stations_list[1]['coords']]))
            continue
        
        # Résoudre le TSP OUVERT pour trouver l'ordre optimal
        tour = solve_tsp_open(stations_list)
        
        # Créer les liens entre stations consécutives dans le tour (TSP OUVERT)
        for i in range(len(tour) - 1):  # -1 pour éviter le retour au début
            station1 = stations_list[tour[i]]
            station2 = stations_list[tour[i + 1]]
            neighbor_geoms.append(LineString([station1['coords'], station2['coords']]))

    # Créer le GeoDataFrame des voisins seulement si on a des géométries
    if neighbor_geoms:
        neighbors_proj = gpd.GeoDataFrame(geometry=neighbor_geoms, crs="EPSG:2154")
        neighbors_wgs = neighbors_proj.to_crs("EPSG:4326")
    else:
        neighbors_wgs = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")

# Reprojection pour Folium
edges_wgs = edges_comb.to_crs("EPSG:4326")
stations_wgs = stations_proj.to_crs("EPSG:4326")

# Palette couleurs par route
keys = [k for k in stations_wgs['parent_key'].unique() if pd.notna(k)]
n = len(keys)
if n > 0:
    palette = colormaps['tab20'](np.linspace(0, 1, min(n, 20)))  # Limiter à 20 couleurs
    color_dict = {k: mcolors.to_hex(palette[i % 20]) for i, k in enumerate(keys)}
else:
    color_dict = {}

# Création de la carte
if not stations_wgs.empty:
    center_lat = stations_wgs.geometry.y.mean()
    center_lon = stations_wgs.geometry.x.mean()
else:
    # Coordonnées par défaut pour la Normandie si aucune station
    center_lat = 49.0
    center_lon = 0.0

map_folium = folium.Map(location=[center_lat, center_lon], zoom_start=9)

# Ajouter tronçons OSM
if not edges_wgs.empty:
    folium.GeoJson(
        edges_wgs[['geometry', 'ref', 'name']].to_json(),
        name='Routes & Rails',
        style_function=lambda f: {'color': 'red', 'weight': 2, 'opacity': 0.7},
        tooltip=folium.GeoJsonTooltip(fields=['ref', 'name'], aliases=['ref', 'name'])
    ).add_to(map_folium)

# Ajouter liens station→route
link_geoms = []
for _, row in stations_wgs.iterrows():
    dist = row['dist_to_highway_m']
    if pd.isna(dist) or dist >= 1000:
        continue
    osmid_key = row['osmid_key']
    if not osmid_key:
        continue
    osmid = osmid_key[0]
    
    # Chercher le tronçon correspondant
    matching_edges = edges_wgs[edges_wgs['osmid'] == osmid]
    if matching_edges.empty:
        # Si pas trouvé avec osmid exact, chercher dans les listes
        matching_edges = edges_wgs[edges_wgs['osmid'].apply(
            lambda x: isinstance(x, list) and osmid in x if isinstance(x, list) else x == osmid
        )]
    
    if matching_edges.empty:
        continue
        
    edge_geom = matching_edges.geometry.iloc[0]
    p_st, p_on = nearest_points(row.geometry, edge_geom)
    link_geoms.append(LineString([(p_st.x, p_st.y), (p_on.x, p_on.y)]))

if link_geoms:
    links_wgs = gpd.GeoDataFrame(geometry=link_geoms, crs='EPSG:4326')
    folium.GeoJson(
        links_wgs.to_json(),
        name='Liens station→route (<1 km)',
        style_function=lambda f: {'color': 'green', 'weight': 1, 'dashArray': '5,5'}
    ).add_to(map_folium)

# Ajouter stations colorées
for _, row in stations_wgs.iterrows():
    dist = row['dist_to_highway_m']
    if pd.isna(dist) or dist >= 1000:
        continue
    pk = row['parent_key']
    station_color = color_dict.get(pk, '#000000')
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        color=station_color,
        fill=True,
        fill_color=station_color,
        fill_opacity=0.8,
        popup=(
            f"<b>Station</b>: {row[name_col]}<br>"
            f"<b>Route</b>: {pk}<br>"
            f"<b>Dist (m)</b>: {dist:.1f}"
        )
    ).add_to(map_folium)

# Ajouter liens voisins seulement s'il y en a
if not neighbors_wgs.empty:
    folium.GeoJson(
        neighbors_wgs.to_json(),
        name='Liens stations voisines',
        style_function=lambda f: {'color': 'blue', 'weight': 3, 'opacity': 0.6}
    ).add_to(map_folium)

# Ajouter légende flottante
if color_dict:
    legend_html = (
        '<div style="position:fixed; bottom:50px; left:50px; '  
        'background:white; border:2px solid grey; padding:8px; max-height:300px; '  
        'overflow-y:auto; font-size:12px; z-index:9999;">'
        '<b>Routes (couleurs)</b><br>'
    )
    for k, c in color_dict.items():
        legend_html += (
            f'<i style="background:{c}; width:12px; height:12px; '  
            'display:inline-block; margin-right:6px;"></i>{k}<br>'
        )
    legend_html += (
        '<hr style="margin:4px 0;">'
        '<i style="background:green; width:12px; height:2px; '  
        'display:inline-block; margin-right:6px;"></i>Station→Route<br>'
        '<i style="background:blue; width:12px; height:2px; '  
        'display:inline-block; margin-right:6px;"></i>Stations voisines'
        '</div>'
    )
    map_folium.get_root().html.add_child(folium.Element(legend_html))

# Contrôle des calques et sauvegarde
folium.LayerControl(collapsed=False).add_to(map_folium)
map_folium.save("map_stations_routes_full.html")
print("Carte enregistrée : map_stations_routes_full.html")