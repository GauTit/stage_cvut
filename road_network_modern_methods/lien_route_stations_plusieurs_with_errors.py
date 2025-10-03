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
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error

# ==================== CONFIGURATION ET PARAMÈTRES ====================

# Chemins des fichiers
input_file = '2023_T4_sites_Metropole.csv'
output_file = 'stations_normandie_orange.csv'

string_dep = "Normandie"
string_dep_arrange = string_dep + ", France"

# Paramètres
buffer_error_atour = 1000  # Buffer de couverture en mètres (1km)
MAX_NEIGHBOR_DISTANCE = 20000  # Distance max entre stations voisines (20km)

# Configuration OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True

print(f"===== 🏁 DÉBUT ANALYSE COUVERTURE ROUTIÈRE - {string_dep} =====")

# ==================== PRÉPARATION DES DONNÉES STATIONS ====================

print("📋 Préparation des données de stations...")

# Lecture et filtrage du CSV original
df = pd.read_csv(input_file, sep=';')
filtered = df[(df['nom_reg'] == string_dep) & (df['nom_op'] == 'Orange')]
filtered.to_csv(output_file, index=False, sep=';')

# Charger les stations filtrées
stations_df = pd.read_csv(
    output_file,
    sep=None,
    engine='python',
    decimal=','
)

print(f"✅ Stations trouvées: {len(stations_df)}")

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

# ==================== TÉLÉCHARGEMENT DES DONNÉES OSM ====================

print("🌐 Téléchargement des données OSM...")

# 1. TOUTES LES ROUTES PRINCIPALES DE LA RÉGION (pour calcul couverture globale)
road_filter = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link"]'
print(f"📡 Téléchargement routes principales: {string_dep_arrange}")

G_road_global = ox.graph_from_place(
    string_dep_arrange,
    network_type=None,
    custom_filter=road_filter,
    retain_all=True
)
nodes_road_global, edges_road_global = ox.graph_to_gdfs(G_road_global)
edges_road_global = edges_road_global.to_crs("EPSG:2154")

print(f"✅ Routes principales téléchargées: {len(edges_road_global)}")
print(f"✅ Longueur totale: {edges_road_global.length.sum()/1000:.1f} km")

# 2. ROUTES LOCALES (pour rattachement des stations - zone limitée)
# Définir zone d'intérêt autour des stations (buffer 10 km)
minx, miny, maxx, maxy = stations_proj.total_bounds
buffer = 10000  # mètres
roi = box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

# Télécharger routes dans la zone d'intérêt pour rattachement
G_road_local = ox.graph_from_place(
    string_dep_arrange,
    network_type=None,
    custom_filter=road_filter,
    retain_all=True
)
nodes_road_local, edges_road_local = ox.graph_to_gdfs(G_road_local)
edges_road_local = edges_road_local.to_crs("EPSG:2154")
edges_road_local = edges_road_local[edges_road_local.intersects(roi)].copy()

# 3. VOIES FERRÉES (optionnel)
try:
    rail_filter = '["railway"="rail"]'
    G_rail = ox.graph_from_place(
        string_dep_arrange,
        network_type=None,
        custom_filter=rail_filter,
        retain_all=True
    )
    nodes_rail, edges_rail = ox.graph_to_gdfs(G_rail)
    edges_rail = edges_rail.to_crs("EPSG:2154")
    edges_rail = edges_rail[edges_rail.intersects(roi)].copy()
    print(f"✅ Voies ferrées trouvées: {len(edges_rail)}")
except:
    edges_rail = gpd.GeoDataFrame(columns=['geometry', 'osmid'], crs="EPSG:2154")
    print("⚠️ Aucune voie ferrée trouvée")

# ==================== RATTACHEMENT DES STATIONS AUX ROUTES ====================

print("🔗 Rattachement des stations aux routes...")

# Combiner routes et rails pour rattachement
def subset_osm(edges):
    cols = ['geometry', 'osmid']
    for extra in ['ref', 'name']:
        if extra in edges.columns:
            cols.append(extra)
    return edges[cols].copy()

edges_for_matching = pd.concat([
    subset_osm(edges_road_local), 
    subset_osm(edges_rail)
], ignore_index=True)

# Créer buffer autour des stations pour trouver routes proches
stations_buffered = stations_proj.copy()
stations_buffered['geometry'] = stations_buffered.geometry.buffer(1000)  # Buffer de 1km

# Spatial join pour toutes les routes dans le buffer
stations_all_routes = gpd.sjoin(
    stations_buffered,
    edges_for_matching,
    how='left',
    predicate='intersects'
)

# Calculer distances réelles (pas sur le buffer)
stations_all_routes['dist_to_highway_m'] = stations_all_routes.apply(
    lambda row: stations_proj.loc[row.name, 'geometry'].distance(
        edges_for_matching.loc[row.index_right, 'geometry']
    ) if pd.notna(row.get('index_right')) else np.nan,
    axis=1
)

# Filtrer les relations à moins de 1 km
stations_all_routes = stations_all_routes[stations_all_routes['dist_to_highway_m'] < 1000]

# Remettre la géométrie originale des stations
stations_all_routes['geometry'] = stations_all_routes.apply(
    lambda row: stations_proj.loc[row.name, 'geometry'], axis=1
)

if stations_all_routes.empty:
    print("❌ ERREUR: Aucune station trouvée à moins de 1 km d'une route!")
    exit()

# Renommer et nettoyer
stations_all_routes = stations_all_routes.rename(columns={'osmid': 'nearest_osmid'}).drop(columns=['index_right'])

print(f"✅ Relations station-route trouvées: {len(stations_all_routes)}")
print(f"✅ Stations uniques rattachées: {len(stations_all_routes.index.unique())}")

# ==================== CRÉATION DES CLÉS DE REGROUPEMENT ====================

print("🔑 Création des clés de regroupement...")

# Gestion des osmids multiples
def create_osmid_key(osmid):
    if isinstance(osmid, list):
        return tuple(sorted(osmid))
    if pd.isna(osmid):
        return None
    return (osmid,)

stations_all_routes['osmid_key'] = stations_all_routes['nearest_osmid'].apply(create_osmid_key)

# Assurer la présence des colonnes
for col in ['ref', 'name']:
    if col not in stations_all_routes.columns:
        stations_all_routes[col] = np.nan

# Création clé parent (ref > name > osmid)
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

# ==================== ALGORITHMES DE RECONSTRUCTION ====================

def solve_tsp_open_with_distance_limit(points, max_distance):
    """Résoudre TSP OUVERT avec algorithme glouton et limite de distance"""
    if len(points) <= 2:
        return list(range(len(points)))
    
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = points[i]['geometry'].distance(points[j]['geometry'])
    
    # Algorithme du plus proche voisin avec limite
    unvisited = set(range(1, n))
    current = 0
    tour = [current]
    
    while unvisited:
        valid_neighbors = [x for x in unvisited if distances[current][x] <= max_distance]
        if not valid_neighbors:
            break
        nearest = min(valid_neighbors, key=lambda x: distances[current][x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour

def create_neighbor_links_with_limit(points, max_distance):
    """Créer liens entre stations voisines avec limite de distance"""
    if len(points) < 2:
        return []
    
    links = []
    
    if len(points) == 2:
        dist = points[0]['geometry'].distance(points[1]['geometry'])
        if dist <= max_distance:
            links.append(LineString([points[0]['coords'], points[1]['coords']]))
        return links
    
    # Pour plus de 2 stations, utiliser TSP avec limite
    unprocessed = list(range(len(points)))
    
    while unprocessed:
        start_idx = unprocessed[0]
        current_points = [points[i] for i in unprocessed]
        local_to_global = {i: unprocessed[i] for i in range(len(current_points))}
        
        tour = solve_tsp_open_with_distance_limit(current_points, max_distance)
        
        # Créer liens pour ce segment
        for i in range(len(tour) - 1):
            idx1 = local_to_global[tour[i]]
            idx2 = local_to_global[tour[i + 1]]
            
            dist = points[idx1]['geometry'].distance(points[idx2]['geometry'])
            if dist <= max_distance:
                links.append(LineString([points[idx1]['coords'], points[idx2]['coords']]))
        
        processed_global_indices = [local_to_global[i] for i in tour]
        for idx in processed_global_indices:
            unprocessed.remove(idx)
        
        if len(unprocessed) == len(current_points):
            unprocessed.pop(0)
    
    return links

# ==================== CRÉATION DES LIENS ENTRE STATIONS ====================

print("🔗 Création des liens entre stations voisines...")

neighbor_geoms = []
valid = stations_all_routes.dropna(subset=['dist_to_highway_m'])
valid = valid[valid['dist_to_highway_m'] < 1000]

total_links_created = 0

for pk, group in valid.groupby('parent_key'):
    if len(group) < 2:
        continue
    
    # Créer liste des stations uniques
    stations_dict = {}
    for idx, row in group.iterrows():
        station_id = (row.geometry.x, row.geometry.y)
        if station_id not in stations_dict:
            stations_dict[station_id] = {
                'idx': idx,
                'geometry': row.geometry,
                'coords': (row.geometry.x, row.geometry.y)
            }
    
    stations_list = list(stations_dict.values())
    
    # Créer liens avec limite de distance
    route_links = create_neighbor_links_with_limit(stations_list, MAX_NEIGHBOR_DISTANCE)
    links_created = len(route_links)
    total_links_created += links_created
    neighbor_geoms.extend(route_links)

print(f"✅ Total liens créés: {total_links_created} (limite: {MAX_NEIGHBOR_DISTANCE/1000} km)")

# ==================== CALCUL DE LA COUVERTURE GLOBALE RÉELLE ====================

print("\n===== 📊 CALCUL DE LA COUVERTURE GLOBALE RÉELLE =====")

if neighbor_geoms:
    # Créer GeoDataFrame des liens reconstruits
    neighbors_proj = gpd.GeoDataFrame(geometry=neighbor_geoms, crs="EPSG:2154")
    
    # Créer buffer global unique (évite double comptage)
    all_links_union = neighbors_proj.union_all()  # Correction: nouvelle méthode
    global_coverage_buffer = all_links_union.buffer(buffer_error_atour)
    
    # Convertir buffer en GeoDataFrame
    buffer_gdf = gpd.GeoDataFrame([1], geometry=[global_coverage_buffer], crs="EPSG:2154")
    
    # Intersection avec toutes les routes principales
    covered_portions = gpd.overlay(edges_road_global, buffer_gdf, how='intersection')
    
    # Calculer longueurs
    total_length_km = edges_road_global.length.sum() / 1000
    covered_length_km = covered_portions.length.sum() / 1000
    
    # Pourcentage de couverture réel
    real_coverage_percentage = (covered_length_km / total_length_km) * 100
    
    print(f"📏 Longueur totale routes principales {string_dep}: {total_length_km:.1f} km")
    print(f"📏 Longueur couverte (buffer {buffer_error_atour}m): {covered_length_km:.1f} km")
    print(f"🎯 COUVERTURE RÉELLE: {real_coverage_percentage:.2f}%")
    
    # Statistiques détaillées
    nb_routes_totales = len(edges_road_global)
    
    # Compter segments avec couverture (gestion des osmid en liste)
    if 'osmid' in covered_portions.columns and not covered_portions.empty:
        # Gérer les osmid qui peuvent être des listes
        try:
            # Essayer comptage direct d'abord
            routes_avec_couverture = covered_portions['osmid'].nunique()
        except (TypeError, ValueError):
            # Si erreur (osmid en liste), compter manuellement
            unique_osmids = set()
            for osmid in covered_portions['osmid']:
                if isinstance(osmid, list):
                    unique_osmids.update(osmid)
                else:
                    unique_osmids.add(osmid)
            routes_avec_couverture = len(unique_osmids)
    else:
        routes_avec_couverture = len(covered_portions)
    
    print(f"📊 Segments routiers totaux: {nb_routes_totales}")
    print(f"📊 Segments avec couverture: {routes_avec_couverture}")
    print(f"📊 % segments touchés: {(routes_avec_couverture/nb_routes_totales)*100:.1f}%")
    
else:
    print("❌ Aucun lien reconstruit - couverture = 0%")
    real_coverage_percentage = 0.0
    total_length_km = edges_road_global.length.sum() / 1000
    covered_length_km = 0.0
    covered_portions = gpd.GeoDataFrame()

# ==================== CRÉATION DE LA CARTE INTERACTIVE ====================

print("\n🗺️ Création de la carte interactive...")

# Reprojections pour Folium
edges_global_wgs = edges_road_global.to_crs("EPSG:4326")
stations_wgs = valid.to_crs("EPSG:4326")

# Centre de la carte
center_lat = edges_global_wgs.geometry.centroid.y.mean()
center_lon = edges_global_wgs.geometry.centroid.x.mean()

# Créer carte
map_folium = folium.Map(location=[center_lat, center_lon], zoom_start=8)

# 1. TOUTES LES ROUTES PRINCIPALES (rouge)
folium.GeoJson(
    edges_global_wgs[['geometry']].to_json(),
    name=f'Toutes routes principales {string_dep}',
    style_function=lambda f: {
        'color': 'red', 
        'weight': 2, 
        'opacity': 0.5
    }
).add_to(map_folium)

# 2. LIENS RECONSTRUCTION (bleu)
if neighbor_geoms:
    neighbors_wgs = neighbors_proj.to_crs("EPSG:4326")
    folium.GeoJson(
        neighbors_wgs.to_json(),
        name='Routes reconstruites (stations)',
        style_function=lambda f: {
            'color': 'blue', 
            'weight': 3, 
            'opacity': 0.8
        }
    ).add_to(map_folium)

# 3. ZONE DE COUVERTURE GLOBALE (vert)
if neighbor_geoms:
    buffer_gdf_wgs = buffer_gdf.to_crs("EPSG:4326")
    folium.GeoJson(
        buffer_gdf_wgs.to_json(),
        name=f'Zone de couverture ({buffer_error_atour}m)',
        style_function=lambda f: {
            'color': 'green',
            'weight': 1,
            'opacity': 0.4,
            'fill': True,
            'fillColor': 'green',
            'fillOpacity': 0.15
        }
    ).add_to(map_folium)

# 4. PORTIONS EFFECTIVEMENT COUVERTES (jaune)
if not covered_portions.empty:
    covered_portions_wgs = covered_portions.to_crs("EPSG:4326")
    folium.GeoJson(
        covered_portions_wgs[['geometry']].to_json(),
        name='Routes effectivement couvertes',
        style_function=lambda f: {
            'color': 'yellow',
            'weight': 4,
            'opacity': 0.9
        }
    ).add_to(map_folium)

# 5. STATIONS (points colorés par route)
if not stations_wgs.empty:
    keys = [k for k in stations_wgs['parent_key'].unique() if pd.notna(k)]
    n = len(keys)
    if n > 0:
        palette = colormaps['tab20'](np.linspace(0, 1, min(n, 20)))
        color_dict = {k: mcolors.to_hex(palette[i % 20]) for i, k in enumerate(keys)}
    else:
        color_dict = {}

    for _, row in stations_wgs.iterrows():
        dist = row['dist_to_highway_m']
        if pd.isna(dist) or dist >= 1000:
            continue
        
        pk = row['parent_key']
        station_color = color_dict.get(pk, '#000000')
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color=station_color,
            fill=True,
            fill_color=station_color,
            fill_opacity=0.8,
            popup=(
                f"<b>Station</b>: {row[name_col]}<br>"
                f"<b>Route</b>: {pk}<br>"
                f"<b>Distance route</b>: {dist:.0f}m"
            )
        ).add_to(map_folium)

# 6. LÉGENDE AVEC VALEURS RÉELLES
legend_html = (
    '<div style="position:fixed; bottom:50px; left:50px; '
    'background:white; border:2px solid grey; padding:10px; '
    'font-size:14px; z-index:9999; width:320px;">'
    f'<b>📊 COUVERTURE ROUTES PRINCIPALES {string_dep.upper()}</b><br><br>'
    '<b style="color:red;">━━━</b> Toutes routes principales<br>'
    '<b style="color:yellow;">━━━</b> Portions couvertes<br>'
    '<b style="color:blue;">━━━</b> Routes reconstruites<br>'
    '<b style="color:green;">▓▓▓</b> Zone de couverture<br>'
    '<b>●</b> Stations (colorées par route)<br><br>'
    '<b>📈 RÉSULTATS:</b><br>'
    f'🎯 <b>Couverture: {real_coverage_percentage:.2f}%</b><br>'
    f'📏 Total: <b>{total_length_km:.1f} km</b><br>'
    f'📏 Couvert: <b>{covered_length_km:.1f} km</b><br>'
    f'🔗 Buffer: <b>{buffer_error_atour}m</b><br>'
    f'📱 Stations: <b>{len(stations_wgs)}</b><br>'
    '</div>'
)
map_folium.get_root().html.add_child(folium.Element(legend_html))

# Contrôle des calques
folium.LayerControl(collapsed=False).add_to(map_folium)

# Sauvegarde
map_filename = f"map_coverage_{string_dep.lower()}.html"
map_folium.save(map_filename)
print(f"✅ Carte sauvegardée: {map_filename}")

# ==================== EXPORT DES RÉSULTATS ====================

print("\n📄 Export des résultats...")

# Préparer données d'export
stations_export = valid.copy()
stations_export['latitude'] = stations_export.geometry.to_crs("EPSG:4326").y
stations_export['longitude'] = stations_export.geometry.to_crs("EPSG:4326").x
stations_export = stations_export.drop(columns=['geometry'])

# Statistiques par route
route_stats = (
    valid.groupby('parent_key')
    .size()
    .reset_index(name='station_count')
    .sort_values('station_count', ascending=False)
)

# Métriques globales
global_metrics = {
    'region': string_dep,
    'total_stations_input': len(stations_df),
    'stations_matched_to_roads': len(valid),
    'stations_not_matched': len(stations_df) - len(valid),
    'unique_routes_with_stations': len(route_stats),
    'total_road_network_km': total_length_km,
    'covered_length_km': covered_length_km,
    'coverage_percentage': real_coverage_percentage,
    'buffer_radius_m': buffer_error_atour,
    'max_neighbor_distance_m': MAX_NEIGHBOR_DISTANCE,
    'total_reconstruction_links': total_links_created,
    'road_types_analyzed': 'motorway,motorway_link,trunk,trunk_link,primary,primary_link'
}

# Export Excel
excel_filename = f"analysis_coverage_{string_dep.lower()}.xlsx"
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Stations
    stations_export[
        ['latitude','longitude','nearest_osmid','ref','name','dist_to_highway_m','parent_key']
    ].to_excel(writer, sheet_name='Stations', index=False)
    
    # Statistiques routes
    route_stats.to_excel(writer, sheet_name='Routes_Stats', index=False)
    
    # Métriques globales
    global_df = pd.DataFrame([global_metrics])
    global_df.to_excel(writer, sheet_name='Global_Metrics', index=False)

print(f"✅ Fichier Excel sauvegardé: {excel_filename}")

# ==================== RÉSUMÉ FINAL ====================

print(f"\n===== ✅ ANALYSE TERMINÉE - {string_dep.upper()} =====")
print(f"🎯 COUVERTURE GLOBALE: {real_coverage_percentage:.2f}%")
print(f"📏 Réseau routier total: {total_length_km:.1f} km")
print(f"📏 Longueur couverte: {covered_length_km:.1f} km")
print(f"📱 Stations analysées: {len(stations_df)} → {len(valid)} utilisées")
print(f"🔗 Liens de reconstruction: {total_links_created}")
print(f"🗺️ Carte: {map_filename}")
print(f"📊 Données: {excel_filename}")

if real_coverage_percentage < 30:
    print("⚠️ COUVERTURE FAIBLE - Recommandations:")
    print("   • Augmenter le nombre de stations")
    print("   • Réduire MAX_NEIGHBOR_DISTANCE si fragmenté")
    print("   • Vérifier la qualité des données stations")
elif real_coverage_percentage < 70:
    print("✅ COUVERTURE MOYENNE - Bon potentiel d'amélioration")
else:
    print("🏆 EXCELLENTE COUVERTURE - Réseau bien maillé")