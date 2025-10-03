import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, box, LineString
from shapely.ops import nearest_points
from shapely.strtree import STRtree
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

# OPTIMISATION : Calculer distance à la route la plus proche pour TOUTES les stations avec index spatial
print("Calcul des distances aux routes pour toutes les stations...")
print(f"  Nombre de stations : {len(stations_proj)}")
print(f"  Nombre de routes/rails : {len(edges_comb)}")

# Créer un index spatial pour les routes
print("  Création de l'index spatial...")
edge_geometries = edges_comb.geometry.values
edge_tree = STRtree(edge_geometries)

# Préparer les colonnes de résultats
distances = []
osmids = []
refs = []
names = []

# Traitement avec affichage de progression
batch_size = 100
n_stations = len(stations_proj)

for i, (idx, station) in enumerate(stations_proj.iterrows()):
    if i % batch_size == 0:
        print(f"  Traitement : {i}/{n_stations} stations ({i/n_stations*100:.1f}%)")
    
    # Trouver la route la plus proche avec l'index spatial
    nearest_idx = edge_tree.nearest(station.geometry)
    nearest_edge = edges_comb.iloc[nearest_idx]
    
    # Calculer la distance exacte
    dist = station.geometry.distance(nearest_edge.geometry)
    
    distances.append(dist)
    osmids.append(nearest_edge['osmid'])
    refs.append(nearest_edge.get('ref', None))
    names.append(nearest_edge.get('name', None))

print(f"  Traitement : {n_stations}/{n_stations} stations (100.0%)")

# Assigner les valeurs
stations_proj['dist_to_highway_m'] = distances
stations_proj['nearest_osmid'] = osmids
stations_proj['ref'] = refs
stations_proj['name'] = names

print("✅ Calcul des distances terminé!")

# Gestion des osmids multiples (liste -> tuple)
def create_osmid_key(osmid):
    if isinstance(osmid, list):
        return tuple(sorted(osmid))
    if pd.isna(osmid):
        return None
    return (osmid,)

stations_proj['osmid_key'] = stations_proj['nearest_osmid'].apply(create_osmid_key)

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
    if row['osmid_key']:
        return f"osmid:{row['osmid_key']}"
    return "no_route"

stations_proj['parent_key'] = stations_proj.apply(create_parent_key, axis=1)

# Statistiques
stations_near_roads = stations_proj[stations_proj['dist_to_highway_m'] < 1000]
stations_far_from_roads = stations_proj[stations_proj['dist_to_highway_m'] >= 1000]

print(f"\n📊 Statistiques des stations:")
print(f"- Total des stations : {len(stations_proj)}")
print(f"- Stations près des routes (<1km) : {len(stations_near_roads)} ({len(stations_near_roads)/len(stations_proj)*100:.1f}%)")
print(f"- Stations éloignées (>=1km) : {len(stations_far_from_roads)} ({len(stations_far_from_roads)/len(stations_proj)*100:.1f}%)")

# CONSTANTE: Distance maximale entre stations voisines (en mètres)
MAX_NEIGHBOR_DISTANCE = 20000  # 20 km

def solve_tsp_open_with_distance_limit(points, max_distance):
    """
    Résoudre TSP OUVERT avec algorithme glouton et limite de distance
    """
    if len(points) <= 2:
        return list(range(len(points)))
    
    # Matrice des distances
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = points[i]['geometry'].distance(points[j]['geometry'])
    
    # Algorithme du plus proche voisin AVEC limite de distance
    unvisited = set(range(1, n))
    current = 0
    tour = [current]
    
    while unvisited:
        # Trouver le plus proche voisin dans la limite de distance
        valid_neighbors = [x for x in unvisited if distances[current][x] <= max_distance]
        
        if not valid_neighbors:
            # Si aucun voisin dans la limite, arrêter ce segment
            break
            
        nearest = min(valid_neighbors, key=lambda x: distances[current][x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour

def create_neighbor_links_with_limit(points, max_distance):
    """
    Créer des liens entre stations voisines avec limite de distance
    """
    if len(points) < 2:
        return []
    
    links = []
    
    # Pour 2 stations seulement
    if len(points) == 2:
        dist = points[0]['geometry'].distance(points[1]['geometry'])
        if dist <= max_distance:
            links.append(LineString([points[0]['coords'], points[1]['coords']]))
        else:
            print(f"⚠️  Stations trop éloignées ({dist/1000:.1f} km > {max_distance/1000} km) - lien ignoré")
        return links
    
    # Pour plus de 2 stations, utiliser TSP avec limite
    unprocessed = list(range(len(points)))
    
    while unprocessed:
        # Commencer un nouveau segment depuis la première station non traitée
        current_points = [points[i] for i in unprocessed]
        
        # Mapper les indices locaux vers les indices globaux
        local_to_global = {i: unprocessed[i] for i in range(len(current_points))}
        
        # Résoudre TSP pour ce sous-ensemble
        tour = solve_tsp_open_with_distance_limit(current_points, max_distance)
        
        # Créer les liens pour ce segment
        for i in range(len(tour) - 1):
            idx1 = local_to_global[tour[i]]
            idx2 = local_to_global[tour[i + 1]]
            
            dist = points[idx1]['geometry'].distance(points[idx2]['geometry'])
            if dist <= max_distance:
                links.append(LineString([points[idx1]['coords'], points[idx2]['coords']]))
            else:
                print(f"⚠️  Lien ignoré: distance {dist/1000:.1f} km > limite {max_distance/1000} km")
        
        # Retirer les stations traitées
        processed_global_indices = [local_to_global[i] for i in tour]
        for idx in processed_global_indices:
            unprocessed.remove(idx)
        
        # Si il reste des stations non connectées, recommencer
        if len(unprocessed) == len(current_points):
            # Éviter boucle infinie - prendre au moins la première station
            unprocessed.pop(0)
    
    return links

# Calcul des liens entre voisins SUR LA MÊME ROUTE pour TOUTES les stations
print("\n=== Calcul des liens entre stations voisines sur la même route ===")
neighbor_geoms = []
total_links_created = 0

# Grouper par parent_key (route) et créer les liens
for pk, group in stations_proj.groupby('parent_key'):
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
                'coords': (row.geometry.x, row.geometry.y),
                'dist_to_road': row['dist_to_highway_m']
            }
    
    stations_list = list(stations_dict.values())
    
    if len(stations_list) >= 2:
        # Compter les stations proches vs éloignées
        n_near = sum(1 for s in stations_list if s['dist_to_road'] < 1000)
        n_far = sum(1 for s in stations_list if s['dist_to_road'] >= 1000)
        
        print(f"🔄 Route '{pk}' : {len(stations_list)} stations ({n_near} proches, {n_far} éloignées)")
        
        # Créer les liens avec limite de distance
        route_links = create_neighbor_links_with_limit(stations_list, MAX_NEIGHBOR_DISTANCE)
        neighbor_geoms.extend(route_links)
        
        if route_links:
            print(f"   ✅ {len(route_links)} lien(s) créé(s)")
            total_links_created += len(route_links)

print(f"\n📊 Total: {total_links_created} liens créés entre stations voisines")

# Créer le GeoDataFrame des voisins
if neighbor_geoms:
    neighbors_proj = gpd.GeoDataFrame(geometry=neighbor_geoms, crs="EPSG:2154")
    neighbors_wgs = neighbors_proj.to_crs("EPSG:4326")
else:
    neighbors_wgs = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
    print("⚠️  Aucun lien créé entre stations")

# Export Excel
stations_export = stations_proj.copy()
stations_export['latitude'] = stations_export.geometry.to_crs("EPSG:4326").y
stations_export['longitude'] = stations_export.geometry.to_crs("EPSG:4326").x
stations_export = stations_export.drop(columns=['geometry'])

# Statistiques par route
route_stats = (
    stations_proj
    .groupby('parent_key')
    .agg({
        'dist_to_highway_m': ['count', 'mean', 'min', 'max'],
        name_col: 'first'  # Exemple de station sur cette route
    })
    .round(1)
)
route_stats.columns = ['nb_stations', 'dist_moy_m', 'dist_min_m', 'dist_max_m', 'exemple_station']
route_stats = route_stats.sort_values('nb_stations', ascending=False)

output_excel = "stations_et_routes_summary.xlsx"
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    stations_export[
        [name_col, 'latitude','longitude','nearest_osmid','ref','name','dist_to_highway_m','parent_key']
    ].to_excel(writer, sheet_name='Stations', index=False)
    route_stats.to_excel(writer, sheet_name='Route_Stats', index=True)

print(f"\n📁 Fichier Excel généré : {output_excel}")

# Reprojection pour Folium
edges_wgs = edges_comb.to_crs("EPSG:4326")
stations_wgs = stations_proj.to_crs("EPSG:4326")

# Palette couleurs par route
keys = [k for k in stations_wgs['parent_key'].unique() if pd.notna(k) and k != 'no_route']
n = len(keys)
if n > 0:
    palette = colormaps['tab20'](np.linspace(0, 1, min(n, 20)))
    color_dict = {k: mcolors.to_hex(palette[i % 20]) for i, k in enumerate(keys)}
else:
    color_dict = {}

# Création de la carte
center_lat = stations_wgs.geometry.y.mean()
center_lon = stations_wgs.geometry.x.mean()
map_folium = folium.Map(location=[center_lat, center_lon], zoom_start=9)

# Ajouter tronçons OSM
if not edges_wgs.empty:
    folium.GeoJson(
        edges_wgs[['geometry', 'ref', 'name']].to_json(),
        name='Routes & Rails',
        style_function=lambda f: {'color': 'red', 'weight': 2, 'opacity': 0.7},
        tooltip=folium.GeoJsonTooltip(fields=['ref', 'name'], aliases=['ref', 'name'])
    ).add_to(map_folium)

# Ajouter liens station→route (seulement pour les stations proches)
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

# Ajouter toutes les stations avec couleurs selon leur route
for _, row in stations_wgs.iterrows():
    dist = row['dist_to_highway_m']
    pk = row['parent_key']
    
    # Couleur selon la route
    if pk in color_dict:
        station_color = color_dict[pk]
    else:
        station_color = '#808080'  # Gris pour les stations sans route identifiée
    
    # Opacité selon la distance
    if dist < 1000:
        opacity = 0.8
        radius = 6
    else:
        opacity = 0.5
        radius = 4
    
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=radius,
        color=station_color,
        fill=True,
        fill_color=station_color,
        fill_opacity=opacity,
        popup=(
            f"<b>Station</b>: {row[name_col]}<br>"
            f"<b>Route</b>: {pk}<br>"
            f"<b>Dist route (m)</b>: {dist:.1f}<br>"
            f"<b>Type</b>: {'Proche route (<1km)' if dist < 1000 else f'Éloignée ({dist/1000:.1f} km)'}"
        )
    ).add_to(map_folium)

# Ajouter liens voisins
if not neighbors_wgs.empty:
    folium.GeoJson(
        neighbors_wgs.to_json(),
        name=f'Liens stations voisines (même route, <{MAX_NEIGHBOR_DISTANCE/1000} km)',
        style_function=lambda f: {'color': 'blue', 'weight': 3, 'opacity': 0.6}
    ).add_to(map_folium)

# Ajouter légende flottante améliorée
legend_html = (
    '<div style="position:fixed; bottom:50px; left:50px; '  
    'background:white; border:2px solid grey; padding:8px; max-height:400px; '  
    'overflow-y:auto; font-size:12px; z-index:9999;">'
    '<b>Légende</b><br>'
    '<hr style="margin:4px 0;">'
    '<b>Stations par route:</b><br>'
)

# Ajouter les couleurs des routes principales
n_routes_shown = 0
for k, c in sorted(color_dict.items(), key=lambda x: x[0]):
    if n_routes_shown < 15:  # Limiter l'affichage
        legend_html += (
            f'<i style="background:{c}; width:12px; height:12px; '  
            'display:inline-block; margin-right:6px; border-radius:50%;"></i>{k}<br>'
        )
        n_routes_shown += 1

if len(color_dict) > 15:
    legend_html += f'<i>... et {len(color_dict) - 15} autres routes</i><br>'

legend_html += (
    '<hr style="margin:4px 0;">'
    '<b>Distance aux routes:</b><br>'
    '<i style="opacity:0.8;">●</i> Station < 1km<br>'
    '<i style="opacity:0.5;">●</i> Station ≥ 1km<br>'
    '<hr style="margin:4px 0;">'
    '<b>Liens:</b><br>'
    '<i style="background:green; width:12px; height:2px; '  
    'display:inline-block; margin-right:6px;"></i>Station→Route<br>'
    f'<i style="background:blue; width:12px; height:2px; '  
    f'display:inline-block; margin-right:6px;"></i>Voisins même route'
    '</div>'
)

map_folium.get_root().html.add_child(folium.Element(legend_html))

# Contrôle des calques et sauvegarde
folium.LayerControl(collapsed=False).add_to(map_folium)
map_folium.save("map_stations_routes_full.html")
print(f"\n🗺️  Carte enregistrée : map_stations_routes_full.html")

# Statistiques finales
print("\n===== 📊 Résumé des statistiques =====")

# Distances pour toutes les stations
print(f"\n📏 Distances aux routes:")
print(f"  Distance moyenne station → route : {stations_proj['dist_to_highway_m'].mean():.1f} m")
print(f"  Distance médiane station → route : {stations_proj['dist_to_highway_m'].median():.1f} m")
print(f"  Distance max station → route : {stations_proj['dist_to_highway_m'].max():.1f} m")

# Répartition des stations
print(f"\n📌 Répartition des stations:")
print(f"  Nombre total de stations : {len(stations_proj)}")
print(f"  Stations proches des routes (<1km) : {len(stations_near_roads)} ({len(stations_near_roads)/len(stations_proj)*100:.1f}%)")
print(f"  Stations éloignées (≥1km) : {len(stations_far_from_roads)} ({len(stations_far_from_roads)/len(stations_proj)*100:.1f}%)")

# Top routes
print(f"\n🛣️  Top 5 routes avec le plus de stations:")
for i, (route, row) in enumerate(route_stats.head(5).iterrows()):
    print(f"  {i+1}. {route} : {row['nb_stations']} stations (dist moy: {row['dist_moy_m']:.0f}m)")

# Analyse des liens
if neighbor_geoms:
    print(f"\n🔗 Analyse des liens entre voisins:")
    print(f"  Nombre total de liens créés : {len(neighbor_geoms)}")
    print(f"  Limite de distance appliquée : {MAX_NEIGHBOR_DISTANCE/1000} km")
    
    # Calculer la distance moyenne des liens
    link_distances = [link.length for link in neighbor_geoms]
    print(f"  Distance moyenne entre voisins : {np.mean(link_distances):.0f} m")
    print(f"  Distance max entre voisins : {np.max(link_distances):.0f} m")

print("\n✅ Analyse terminée!")