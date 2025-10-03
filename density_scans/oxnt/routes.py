import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- 1. Chargement et préparation des stations ---
df = pd.read_csv('stations_normandie_orange.csv', sep=';')
df['lat'] = df['latitude'].str.replace(',', '.').astype(float)
df['lon'] = df['longitude'].str.replace(',', '.').astype(float)
gdf_stations = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df['lon'], df['lat'])],
    crs='EPSG:4326'
)

# --- 2. Récupération du réseau routier (OSM) ---
# (peut prendre quelques minutes selon la découpe)
G = ox.graph_from_place("Normandie, France", network_type='drive')
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# --- 3. Reprojection en metric (Lambert-93) pour calcul de distance ---
gdf_stations = gdf_stations.to_crs(epsg=2154)
gdf_edges    = gdf_edges.to_crs(epsg=2154)

# --- 4. Calcul de la distance station → route la plus proche ---
gdf_nearest = gpd.sjoin_nearest(
    gdf_stations,
    gdf_edges[['geometry']],
    how='left',
    distance_col='dist_m'
)
# On conserve seulement ce dont on a besoin
gdf_nearest = gdf_nearest[['code_op','nom_op','id_station_anfr','dist_m','geometry']]

# --- 5. Création de la carte interactive ---
# Repassage en WGS84
gdf_nearest = gdf_nearest.to_crs(epsg=4326)
gdf_edges   = gdf_edges.to_crs(epsg=4326)

# Centre de la carte
center = [gdf_nearest.geometry.y.mean(), gdf_nearest.geometry.x.mean()]
m = folium.Map(location=center, zoom_start=8)

# a) Tracé des routes (faible opacité)
for _, edge in gdf_edges.iterrows():
    folium.PolyLine(
        locations=[(y, x) for x, y in edge.geometry.coords],
        weight=1,
        opacity=0.5
    ).add_to(m)

# b) Stations colorées selon la distance (graduation)
# Choix d'une colormap
norm = mcolors.Normalize(vmin=0, vmax=gdf_nearest['dist_m'].quantile(0.95))
cmap = cm.get_cmap('viridis')

for _, row in gdf_nearest.iterrows():
    color = mcolors.to_hex(cmap(norm(row['dist_m'])))
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=f"Dist. route : {row['dist_m']:.1f} m"
    ).add_to(m)

# --- 6. Sauvegarde ---
output_html = 'stations_vs_routes.html'
m.save(output_html)
print(f"Carte enregistrée ici : {output_html}")
