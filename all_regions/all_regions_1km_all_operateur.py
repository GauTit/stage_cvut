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
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False  # Réduire les logs pour la lisibilité

class TelecomAnalyzer:
    def __init__(self, input_file='2023_T4_sites_Metropole.csv'):
        self.input_file = input_file
        self.operators = ['Orange', 'Free Mobile', 'SFR', 'Bouygues Telecom']
        self.results = {}
        self.global_stats = []
        
        # Constantes
        self.MAX_NEIGHBOR_DISTANCE = 20000  # 20 km
        self.BUFFER_DISTANCE = 1000  # 1 km
        
        # Charger les données une seule fois
        print("📂 Chargement des données...")
        self.df = pd.read_csv(input_file, sep=';', decimal=',')
        self.regions = sorted(self.df['nom_reg'].unique())
        print(f"✅ {len(self.df)} stations chargées pour {len(self.regions)} régions")
        
    def create_stations_gdf(self, stations_df):
        """Créer un GeoDataFrame à partir des données de stations"""
        try:
            # Détecter les colonnes de coordonnées
            lat_col = next((col for col in stations_df.columns if 'lat' in col.lower()), None)
            lon_col = next((col for col in stations_df.columns if 'lon' in col.lower() or 'lng' in col.lower()), None)
            
            if lat_col is None or lon_col is None:
                raise ValueError("Colonnes latitude/longitude non trouvées")
            
            # Convertir les coordonnées en format numérique (gérer les virgules françaises)
            stations_clean = stations_df.copy()
            
            # Convertir les colonnes de coordonnées si elles sont en string avec virgules
            for coord_col in [lat_col, lon_col]:
                if stations_clean[coord_col].dtype == 'object':
                    # Remplacer virgules par points et convertir en float
                    stations_clean[coord_col] = stations_clean[coord_col].astype(str).str.replace(',', '.').astype(float)
            
            # Créer GeoDataFrame
            stations_gdf = gpd.GeoDataFrame(
                stations_clean,
                geometry=stations_clean.apply(lambda r: Point(r[lon_col], r[lat_col]), axis=1),
                crs="EPSG:4326"
            )
            return stations_gdf
        except Exception as e:
            print(f"❌ Erreur création GeoDataFrame: {e}")
            return None
    
    def get_road_network(self, region_name, stations_gdf):
        """Télécharger le réseau routier pour une région"""
        try:
            # Projection en Lambert-93
            stations_proj = stations_gdf.to_crs("EPSG:2154")
            
            # Zone d'intérêt avec buffer
            minx, miny, maxx, maxy = stations_proj.total_bounds
            buffer = 10000  # 10 km
            roi = box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
            
            # Télécharger routes principales
            road_filter = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link"]'
            G_road = ox.graph_from_place(
                f"{region_name}, France",
                network_type=None,
                custom_filter=road_filter,
                retain_all=True
            )
            nodes_road, edges_road = ox.graph_to_gdfs(G_road)
            edges_road = edges_road.to_crs("EPSG:2154")
            edges_road = edges_road[edges_road.intersects(roi)].copy()
            
            # Télécharger voies ferrées
            rail_filter = '["railway"="rail"]'
            G_rail = ox.graph_from_place(
                f"{region_name}, France",
                network_type=None,
                custom_filter=rail_filter,
                retain_all=True
            )
            nodes_rail, edges_rail = ox.graph_to_gdfs(G_rail)
            edges_rail = edges_rail.to_crs("EPSG:2154")
            edges_rail = edges_rail[edges_rail.intersects(roi)].copy()
            
            # Combiner routes et rails
            def subset_osm(edges):
                cols = ['geometry', 'osmid']
                for extra in ['ref', 'name']:
                    if extra in edges.columns:
                        cols.append(extra)
                return edges[cols].copy()
            
            edges_comb = pd.concat([subset_osm(edges_road), subset_osm(edges_rail)], ignore_index=True)
            return edges_comb, roi
            
        except Exception as e:
            print(f"❌ Erreur réseau routier pour {region_name}: {e}")
            return None, None
    
    def analyze_stations_routes(self, stations_gdf, edges_comb):
        """Analyser les relations stations-routes"""
        try:
            # Projection
            stations_proj = stations_gdf.to_crs("EPSG:2154")
            
            # Buffer autour des stations
            stations_buffered = stations_proj.copy()
            stations_buffered['geometry'] = stations_buffered.geometry.buffer(self.BUFFER_DISTANCE)
            
            # Spatial join
            stations_all_routes = gpd.sjoin(
                stations_buffered,
                edges_comb,
                how='left',
                predicate='intersects'
            )
            
            # Calculer distances réelles
            stations_all_routes['dist_to_highway_m'] = stations_all_routes.apply(
                lambda row: stations_proj.loc[row.name, 'geometry'].distance(
                    edges_comb.loc[row.index_right, 'geometry']
                ) if pd.notna(row.get('index_right')) else np.nan,
                axis=1
            )
            
            # Filtrer < 1km
            stations_all_routes = stations_all_routes[stations_all_routes['dist_to_highway_m'] < 1000]
            
            # Remettre géométrie originale
            stations_all_routes['geometry'] = stations_all_routes.apply(
                lambda row: stations_proj.loc[row.name, 'geometry'], axis=1
            )
            
            if stations_all_routes.empty:
                return None
            
            # Renommer et nettoyer
            stations_all_routes = stations_all_routes.rename(columns={'osmid': 'nearest_osmid'}).drop(columns=['index_right'])
            
            # Gérer colonnes ref et name
            for col in ['ref', 'name']:
                if col not in stations_all_routes.columns:
                    stations_all_routes[col] = np.nan
            
            # Créer clé parent
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
                return f"osmid:{row['nearest_osmid']}"
            
            stations_all_routes['parent_key'] = stations_all_routes.apply(create_parent_key, axis=1)
            
            return stations_all_routes
            
        except Exception as e:
            print(f"❌ Erreur analyse stations-routes: {e}")
            return None
    
    def calculate_stats(self, stations_data, total_stations):
        """Calculer les statistiques pour une combinaison région/opérateur"""
        if stations_data is None or stations_data.empty:
            return {
                'total_stations': total_stations,
                'stations_rattachees': 0,
                'pourcentage_rattachees': 0.0,
                'distance_moyenne': np.nan,
                'distance_mediane': np.nan,
                'nb_routes': 0
            }
        
        nb_stations_rattachees = stations_data.index.nunique()
        distances = stations_data['dist_to_highway_m'].dropna()
        
        return {
            'total_stations': total_stations,
            'stations_rattachees': nb_stations_rattachees,
            'pourcentage_rattachees': (nb_stations_rattachees / total_stations * 100) if total_stations > 0 else 0,
            'distance_moyenne': distances.mean() if not distances.empty else np.nan,
            'distance_mediane': distances.median() if not distances.empty else np.nan,
            'nb_routes': len(stations_data['parent_key'].unique()) if 'parent_key' in stations_data.columns else 0
        }
    
    def save_individual_files(self, region, operator, stations_data, stats):
        """Sauvegarder les fichiers individuels pour chaque région/opérateur"""
        try:
            # Créer dossier de sortie
            output_dir = f"resultats_{region.replace(' ', '_')}_{operator.replace(' ', '_')}"
            os.makedirs(output_dir, exist_ok=True)
            
            if stations_data is not None and not stations_data.empty:
                # Préparer données export
                stations_export = stations_data.copy()
                stations_export['latitude'] = stations_export.geometry.to_crs("EPSG:4326").y
                stations_export['longitude'] = stations_export.geometry.to_crs("EPSG:4326").x
                stations_export = stations_export.drop(columns=['geometry'])
                
                # Statistiques par route
                route_stats = (
                    stations_data.groupby('parent_key')
                    .size()
                    .reset_index(name='station_count')
                    .sort_values('station_count', ascending=False)
                )
                
                # Fichier Excel
                excel_file = os.path.join(output_dir, f"stations_et_routes_{region}_{operator}.xlsx")
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    stations_export[
                        ['latitude','longitude','nearest_osmid','ref','name','dist_to_highway_m','parent_key']
                    ].to_excel(writer, sheet_name='Stations', index=False)
                    route_stats.to_excel(writer, sheet_name='Route_Stats', index=False)
                    
                    # Ajouter feuille statistiques générales
                    stats_df = pd.DataFrame([stats])
                    stats_df.to_excel(writer, sheet_name='Statistiques', index=False)
                
                return excel_file
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde {region}-{operator}: {e}")
        
        return None
    
    def process_region_operator(self, region, operator):
        """Traiter une combinaison région/opérateur"""
        try:
            # Filtrer données
            filtered_data = self.df[(self.df['nom_reg'] == region) & (self.df['nom_op'] == operator)]
            
            if filtered_data.empty:
                stats = self.calculate_stats(None, 0)
                return stats, None
            
            total_stations = len(filtered_data)
            
            # Créer GeoDataFrame
            stations_gdf = self.create_stations_gdf(filtered_data)
            if stations_gdf is None:
                stats = self.calculate_stats(None, total_stations)
                return stats, None
            
            # Obtenir réseau routier (cache automatique par osmnx)
            edges_comb, roi = self.get_road_network(region, stations_gdf)
            if edges_comb is None:
                stats = self.calculate_stats(None, total_stations)
                return stats, None
            
            # Analyser relations stations-routes
            stations_data = self.analyze_stations_routes(stations_gdf, edges_comb)
            
            # Calculer statistiques
            stats = self.calculate_stats(stations_data, total_stations)
            
            # Sauvegarder fichiers individuels
            self.save_individual_files(region, operator, stations_data, stats)
            
            return stats, stations_data
            
        except Exception as e:
            print(f"❌ Erreur traitement {region}-{operator}: {e}")
            stats = self.calculate_stats(None, len(self.df[(self.df['nom_reg'] == region) & (self.df['nom_op'] == operator)]))
            return stats, None
    
    def run_analysis(self):
        """Lancer l'analyse complète"""
        print(f"\n🚀 Début de l'analyse pour {len(self.regions)} régions et {len(self.operators)} opérateurs")
        print(f"📊 Total: {len(self.regions) * len(self.operators)} combinaisons à traiter\n")
        
        total_combinations = len(self.regions) * len(self.operators)
        processed = 0
        
        for region in self.regions:
            self.results[region] = {}
            
            for operator in self.operators:
                processed += 1
                progress = (processed / total_combinations) * 100
                
                print(f"⏳ [{processed:3d}/{total_combinations}] ({progress:5.1f}%) {region} - {operator}... ", end="")
                
                stats, data = self.process_region_operator(region, operator)
                self.results[region][operator] = {
                    'stats': stats,
                    'data': data
                }
                
                # Ajouter aux stats globales
                stats_entry = {
                    'region': region,
                    'operator': operator,
                    **stats
                }
                self.global_stats.append(stats_entry)
                
                # Afficher résultat
                if stats['total_stations'] > 0:
                    print(f"✅ {stats['stations_rattachees']}/{stats['total_stations']} stations ({stats['pourcentage_rattachees']:.1f}%)")
                else:
                    print("⚪ Pas de données")
        
        print(f"\n🎉 Analyse terminée ! Génération des fichiers consolidés...")
        self.save_consolidated_results()
    
    def save_consolidated_results(self):
        """Sauvegarder les résultats consolidés"""
        try:
            # DataFrame des statistiques globales
            global_df = pd.DataFrame(self.global_stats)
            
            # Créer fichier Excel consolidé
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            consolidated_file = f"analyse_telecom_consolidee_{timestamp}.xlsx"
            
            with pd.ExcelWriter(consolidated_file, engine='openpyxl') as writer:
                # Feuille statistiques globales
                global_df.to_excel(writer, sheet_name='Statistiques_Globales', index=False)
                
                # Feuille résumé par région
                region_summary = global_df.groupby('region').agg({
                    'total_stations': 'sum',
                    'stations_rattachees': 'sum',
                    'pourcentage_rattachees': 'mean',
                    'distance_moyenne': 'mean',
                    'distance_mediane': 'mean'
                }).round(2)
                region_summary.to_excel(writer, sheet_name='Resume_Regions')
                
                # Feuille résumé par opérateur
                operator_summary = global_df.groupby('operator').agg({
                    'total_stations': 'sum',
                    'stations_rattachees': 'sum',
                    'pourcentage_rattachees': 'mean',
                    'distance_moyenne': 'mean',
                    'distance_mediane': 'mean'
                }).round(2)
                operator_summary.to_excel(writer, sheet_name='Resume_Operateurs')
                
                # Top 10 des meilleures combinaisons
                top_combinations = global_df.nlargest(10, 'pourcentage_rattachees')[
                    ['region', 'operator', 'total_stations', 'stations_rattachees', 'pourcentage_rattachees', 'distance_moyenne']
                ]
                top_combinations.to_excel(writer, sheet_name='Top_10_Combinaisons', index=False)
            
            print(f"📁 Fichier consolidé créé: {consolidated_file}")
            
            # Afficher résumé final
            self.print_final_summary(global_df)
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde consolidée: {e}")
    
    def print_final_summary(self, global_df):
        """Afficher le résumé final"""
        print("\n" + "="*60)
        print("📊 RÉSUMÉ FINAL DE L'ANALYSE")
        print("="*60)
        
        total_stations = global_df['total_stations'].sum()
        total_rattachees = global_df['stations_rattachees'].sum()
        taux_global = (total_rattachees / total_stations * 100) if total_stations > 0 else 0
        
        print(f"🏢 Total stations analysées: {total_stations:,}")
        print(f"✅ Total stations rattachées: {total_rattachees:,}")
        print(f"📈 Taux de rattachement global: {taux_global:.2f}%")
        print(f"📏 Distance moyenne globale: {global_df['distance_moyenne'].mean():.1f}m")
        
        print(f"\n🥇 TOP 3 RÉGIONS (par taux de rattachement):")
        top_regions = global_df.groupby('region')['pourcentage_rattachees'].mean().nlargest(3)
        for i, (region, taux) in enumerate(top_regions.items(), 1):
            print(f"   {i}. {region}: {taux:.2f}%")
        
        print(f"\n📱 TOP 3 OPÉRATEURS (par taux de rattachement):")
        top_operators = global_df.groupby('operator')['pourcentage_rattachees'].mean().nlargest(3)
        for i, (operator, taux) in enumerate(top_operators.items(), 1):
            print(f"   {i}. {operator}: {taux:.2f}%")
        
        print("="*60)

# Lancement de l'analyse
if __name__ == "__main__":
    analyzer = TelecomAnalyzer('2023_T4_sites_Metropole.csv')
    analyzer.run_analysis()