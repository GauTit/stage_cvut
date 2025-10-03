import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, box
import pandas as pd
import folium
import numpy as np
import time

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
# Détecter la colonne nom
name_col = next(
    (col for col in stations_df.columns if col.lower() not in ['latitude','longitude','lat','lon','lng']),
    stations_df.columns[0]
)
stations_gdf = gpd.GeoDataFrame(
    stations_df,
    geometry=stations_df.apply(lambda r: Point(r.longitude, r.latitude), axis=1),
    crs="EPSG:4326"
)

# Projection en Lambert-93
stations_proj = stations_gdf.to_crs("EPSG:2154")
# Définir la zone d'intérêt (buffer 10 km)
minx, miny, maxx, maxy = stations_proj.total_bounds
buffer = 10000  # en mètres
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
# Projeter et filtrer spatialement
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
# Projeter et filtrer spatialement
edges_rail = edges_rail.to_crs("EPSG:2154")
edges_rail = edges_rail[edges_rail.intersects(roi)].copy()

# 3. Calcul des distances et association du segment le plus proche (méthode vectorisée)
# Fusionner routes et rails pour le join
edges_comb = pd.concat([edges_road[['geometry','osmid']], edges_rail[['geometry','osmid']]], ignore_index=True)
# GeoDataFrame pour jointure
edges_for_join = edges_comb.copy()

# Spatial join nearest pour calculer la distance exacte
stations_joined = gpd.sjoin_nearest(
    stations_proj,
    edges_for_join,
    how='left',
    distance_col='dist_to_highway_m'
)
# Renommer la colonne d'OSMID et nettoyer l'index
stations_proj = stations_joined.rename(columns={'osmid': 'nearest_osmid'})
stations_proj = stations_proj.drop(columns=['index_right'])

# 4. Définir et exclure les grandes villes
# Liste des grandes villes de Normandie
large_cities = [

]

city_polys = []
for city in large_cities:
    try:
        city_gdf = ox.geocode_to_gdf(city)
        city_poly = city_gdf.unary_union
        # projeté en Lambert-93
        city_poly = gpd.GeoSeries([city_poly], crs="EPSG:4326").to_crs("EPSG:2154").iloc[0]
        city_polys.append(city_poly)
    except Exception as e:
        print(f"Impossible de récupérer la géométrie de {city} : {e}")
# Union de toutes les polygones urbains
if city_polys:
    urban_union = gpd.GeoSeries(city_polys, crs="EPSG:2154").unary_union
    stations_proj['in_large_city'] = stations_proj.geometry.within(urban_union)
else:
    stations_proj['in_large_city'] = False

# 5. Statistiques utiles pour stations hors grandes villes
stations_nonurban = stations_proj[~stations_proj['in_large_city']]
total_nonurban = len(stations_nonurban)
under_1k_nonurban = (stations_nonurban.dist_to_highway_m <= 1000).sum()
pct_under_1k_nonurban = under_1k_nonurban / total_nonurban * 100 if total_nonurban > 0 else 0
print(f"Nombre total de stations (hors grandes villes) : {total_nonurban}")
print(f"Pourcentage de stations à moins de 1 km (hors grandes villes) : {pct_under_1k_nonurban:.1f}%")

# 6. Export CSV Statistiques utiles Statistiques utiles

total = len(stations_proj)
under_1k = (stations_proj.dist_to_highway_m <= 1000).sum()
pct_under_1k = under_1k / total * 100
print(f"Nombre total de stations : {total}")
print(f"Pourcentage à moins de 1 km : {pct_under_1k:.1f}%")

# 5. Export CSV
stations_proj.drop(columns=['geometry'], errors='ignore') \
    .to_csv("stations_nearest_combined.csv", index=False)

# 6. Carte Folium
center = [stations_gdf.geometry.y.mean(), stations_gdf.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=9)

# Couches séparées
# Routes
edges_road_geo = edges_road.to_crs("EPSG:4326")
folium.GeoJson(
    edges_road_geo,
    name="Routes",
    style_function=lambda feat: {"color": "blue", "weight": 2, "opacity": 0.6}
).add_to(m)
# Voies ferrées
edges_rail_geo = edges_rail.to_crs("EPSG:4326")
folium.GeoJson(
    edges_rail_geo,
    name="Voies Ferrées",
    style_function=lambda feat: {"color": "black", "weight": 3, "opacity": 0.8}
).add_to(m)

# Stations
for _, row in stations_proj.to_crs("EPSG:4326").iterrows():
    prox = row.dist_to_highway_m <= 1000
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        color='green' if prox else 'red',
        fill=True,
        fill_color='green' if prox else 'red',
        fill_opacity=0.7,
        popup=f"{row[name_col]}<br>Dist: {row.dist_to_highway_m:.0f} m"
    ).add_to(m)

folium.LayerControl().add_to(m)
m.save("stations_combined_map.html")
print("Carte interactive enregistrée sous 'stations_combined_map.html'")
