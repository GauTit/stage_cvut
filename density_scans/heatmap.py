import pandas as pd
import folium
from folium.plugins import HeatMap

# Lire le CSV filtré
df = pd.read_csv('stations_normandie_orange.csv', sep=';')

# Convertir latitude/longitude de chaîne "49,18056" en float
df['latitude'] = df['latitude'].str.replace(',', '.').astype(float)
df['longitude'] = df['longitude'].str.replace(',', '.').astype(float)

# Calculer le centre de la carte
center = [df['latitude'].mean(), df['longitude'].mean()]

# Créer la carte centrée sur la Normandie
m = folium.Map(location=center, zoom_start=8)

# Préparer les données pour la heatmap
heat_data = df[['latitude', 'longitude']].values.tolist()

# Ajouter la couche HeatMap
HeatMap(heat_data, radius=15, blur=25).add_to(m)

# Sauvegarder la carte dans un fichier HTML
output_html = './density_scans/heatmap_normandie_orange.html'
m.save(output_html)