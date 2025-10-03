import pandas as pd
import os
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RegionalCoverageAnalyzer:
    """Analyseur rapide de couverture - stats uniquement"""
    
    def __init__(self, input_file, output_dir="stats_couverture"):
        self.input_file = input_file
        self.output_dir = output_dir
        
        # Créer le dossier de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration OSMnx
        ox.settings.use_cache = True
        ox.settings.log_console = False
        
        # Charger le fichier national avec gestion du format français
        print(f"Chargement du fichier national: {input_file}")
        self.df_national = pd.read_csv(input_file, sep=';', decimal=',', dtype=str)
        
        # Convertir les coordonnées en format numérique
        self._fix_coordinates()
        
        print(f"Total stations chargées: {len(self.df_national)}")
        
        # Obtenir la liste des régions disponibles
        self.regions = self.df_national['nom_reg'].unique()
        print(f"Régions disponibles: {len(self.regions)}")
    
    def _fix_coordinates(self):
        """Corrige le format des coordonnées (virgule -> point)"""
        coord_columns = []
        for col in self.df_national.columns:
            col_lower = col.lower()
            if any(coord in col_lower for coord in ['lat', 'lon', 'x', 'y']):
                coord_columns.append(col)
        
        print(f"Colonnes de coordonnées détectées: {coord_columns}")
        
        for col in coord_columns:
            if isinstance(self.df_national[col].iloc[0], str):
                self.df_national[col] = self.df_national[col].str.replace(',', '.').astype(float)
    
    def get_region_data(self, region_name, operator='Orange'):
        """Filtrer les données pour une région spécifique"""
        return self.df_national[
            (self.df_national['nom_reg'] == region_name) & 
            (self.df_national['nom_op'] == operator)
        ].copy()
    
    def analyze_region_coverage(self, region_name, operator='Orange'):
        """Analyse rapide de couverture d'une région - STATS SEULEMENT"""
        print(f"\n🔍 Analyse couverture: {region_name}")
        
        # Filtrer les données
        stations_df = self.get_region_data(region_name, operator)
        if len(stations_df) == 0:
            print(f"❌ Aucune station {operator} trouvée pour {region_name}")
            return None
            
        print(f"📊 {len(stations_df)} stations {operator} trouvées")
        
        try:
            # Analyser rapidement la région
            stats = self._quick_process_region(stations_df, region_name)
            return stats
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse de {region_name}: {str(e)}")
            return None
    
    def _quick_process_region(self, stations_df, region_name):
        """Traitement rapide - calcul des stats de couverture uniquement"""
        
        # Détecter automatiquement les colonnes de coordonnées
        lat_col = None
        lon_col = None
        
        for col in stations_df.columns:
            col_lower = col.lower()
            if 'lat' in col_lower and lat_col is None:
                lat_col = col
            elif 'lon' in col_lower and lon_col is None:
                lon_col = col
        
        if lat_col is None or lon_col is None:
            raise ValueError(f"Colonnes latitude/longitude non trouvées. Colonnes disponibles: {list(stations_df.columns)}")
        
        # S'assurer que les coordonnées sont numériques
        if stations_df[lat_col].dtype == 'object':
            stations_df[lat_col] = stations_df[lat_col].astype(str).str.replace(',', '.').astype(float)
        if stations_df[lon_col].dtype == 'object':
            stations_df[lon_col] = stations_df[lon_col].astype(str).str.replace(',', '.').astype(float)
        
        # Créer GeoDataFrame des stations
        stations_gdf = gpd.GeoDataFrame(
            stations_df,
            geometry=stations_df.apply(lambda r: Point(r[lon_col], r[lat_col]), axis=1),
            crs="EPSG:4326"
        )
        
        # Projection en Lambert-93 (mètres)
        stations_proj = stations_gdf.to_crs("EPSG:2154")
        
        # Définir zone d'intérêt (buffer 10 km comme l'original)
        minx, miny, maxx, maxy = stations_proj.total_bounds
        buffer = 10000  # 10km comme l'original
        roi = box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
        
        # Télécharger le réseau routier (même filtre que l'original)
        print("📥 Téléchargement du réseau routier principal...")
        road_filter = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link"]'  # Même filtre que l'original
        
        try:
            G_road = ox.graph_from_place(
                f"{region_name}, France",
                network_type=None,
                custom_filter=road_filter,
                retain_all=True
            )
            nodes_road, edges_road = ox.graph_to_gdfs(G_road)
            edges_road = edges_road.to_crs("EPSG:2154")
            edges_road = edges_road[edges_road.intersects(roi)].copy()
            
            if edges_road.empty:
                raise Exception("Aucune route trouvée")
                
        except Exception as e:
            print(f"⚠️ Erreur téléchargement routes: {e}")
            return self._create_fallback_stats(stations_df, region_name, error="no_roads")
        
        # Simplifier les edges pour les calculs
        edges_simple = edges_road[['geometry']].copy()
        edges_simple['osmid'] = range(len(edges_simple))  # ID simple
        
        # Association stations-routes (version simplifiée)
        print("🔗 Association stations-routes...")
        stations_buffered = stations_proj.copy()
        stations_buffered['geometry'] = stations_buffered.geometry.buffer(1000)  # Buffer de 1km (comme l'original)
        
        # Spatial join
        try:
            stations_routes = gpd.sjoin(
                stations_buffered,
                edges_simple,
                how='left',
                predicate='intersects'
            )
            
            # Vérifier les colonnes créées par sjoin
            print(f"Colonnes après sjoin: {list(stations_routes.columns)}")
            
            # Gérer différentes versions de GeoPandas
            right_index_col = None
            for col in stations_routes.columns:
                if 'index_right' in col or col == 'index_right':
                    right_index_col = col
                    break
            
            if right_index_col is None:
                # Fallback: utiliser l'index des osmid
                stations_routes['temp_right_idx'] = stations_routes.get('osmid', None)
                right_index_col = 'temp_right_idx'
            
            print(f"Utilisation de la colonne d'index: {right_index_col}")
            
            # Calculer les distances réelles (seulement pour les stations qui ont un match)
            valid_matches = stations_routes.dropna(subset=[right_index_col])
            print(f"Matches trouvés: {len(valid_matches)}/{len(stations_routes)}")
            
            if len(valid_matches) > 0:
                print("⚡ Optimisation: calcul des distances par vectorisation...")
                
                # OPTIMISATION: Au lieu de calculer toutes les distances, 
                # ne garder que la distance minimale par station
                unique_stations = valid_matches.index.unique()
                print(f"Stations uniques à traiter: {len(unique_stations)}")
                
                min_distances = {}
                
                # Traitement par batch pour éviter de surcharger la mémoire
                batch_size = 100
                for i in range(0, len(unique_stations), batch_size):
                    batch_stations = unique_stations[i:i+batch_size]
                    
                    if i % 500 == 0:  # Progress indicator
                        print(f"  Traitement stations {i}-{min(i+batch_size, len(unique_stations))}/{len(unique_stations)}")
                    
                    for station_idx in batch_stations:
                        # Récupérer toutes les routes pour cette station
                        station_matches = valid_matches.loc[[station_idx]]
                        original_station = stations_proj.loc[station_idx, 'geometry']
                        
                        min_dist = float('inf')
                        best_osmid = None
                        
                        # Calculer la distance à toutes les routes associées à cette station
                        for _, match_row in station_matches.iterrows():
                            try:
                                if right_index_col == 'temp_right_idx':
                                    # Méthode directe pour toutes les routes
                                    for _, edge in edges_simple.iterrows():
                                        dist = original_station.distance(edge.geometry)
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_osmid = edge.osmid
                                    break  # Une seule fois suffit
                                else:
                                    route_idx = match_row[right_index_col]
                                    if pd.notna(route_idx) and route_idx in edges_simple.index:
                                        route_geom = edges_simple.loc[route_idx, 'geometry']
                                        dist = original_station.distance(route_geom)
                                        if dist < min_dist:
                                            min_dist = dist
                                            best_osmid = match_row.get('osmid', route_idx)
                                            
                            except Exception as e:
                                continue
                        
                        if min_dist < float('inf'):
                            min_distances[station_idx] = {
                                'dist_to_highway_m': min_dist,
                                'osmid': best_osmid
                            }
                
                # Créer le DataFrame final avec seulement les distances minimales
                stations_routes_clean = stations_proj.copy()
                distances = []
                osmids = []
                
                for idx in stations_routes_clean.index:
                    if idx in min_distances:
                        distances.append(min_distances[idx]['dist_to_highway_m'])
                        osmids.append(min_distances[idx]['osmid'])
                    else:
                        distances.append(np.nan)
                        osmids.append(np.nan)
                
                stations_routes = stations_routes_clean
                stations_routes['dist_to_highway_m'] = distances
                stations_routes['osmid'] = osmids
                
                print(f"✅ Distances calculées pour {len([d for d in distances if not pd.isna(d)])} stations")
                
            else:
                print("Aucun match trouvé - utilisation méthode alternative directe")
                # Méthode alternative ultra-rapide : distance minimale par station
                print("⚡ Calcul direct optimisé...")
                
                distances = []
                osmids = []
                
                # Pré-calculer toutes les géométries des routes pour l'optimisation
                route_geometries = list(edges_simple.geometry)
                route_osmids = list(edges_simple.osmid)
                
                batch_size = 50
                for i in range(0, len(stations_proj), batch_size):
                    batch_stations = stations_proj.iloc[i:i+batch_size]
                    
                    if i % 200 == 0:
                        print(f"  Stations {i}-{min(i+batch_size, len(stations_proj))}/{len(stations_proj)}")
                    
                    for idx, station_row in batch_stations.iterrows():
                        min_dist = float('inf')
                        best_osmid = None
                        
                        # Vectorized distance calculation
                        for j, route_geom in enumerate(route_geometries):
                            dist = station_row.geometry.distance(route_geom)
                            if dist < min_dist:
                                min_dist = dist
                                best_osmid = route_osmids[j]
                                
                        distances.append(min_dist if min_dist < float('inf') else np.nan)
                        osmids.append(best_osmid)
                
                stations_routes = stations_proj.copy()
                stations_routes['dist_to_highway_m'] = distances
                stations_routes['osmid'] = osmids
                print(f"✅ Calcul direct: {len([d for d in distances if not pd.isna(d)])} distances calculées")
            
        except Exception as e:
            print(f"⚠️ Erreur spatial join: {e}")
            return self._create_fallback_stats(stations_df, region_name, error="spatial_join_failed")
        
        # Calculer les statistiques de couverture
        stats = self._calculate_quick_stats(stations_routes, len(stations_df), region_name)
        
        print(f"✅ {region_name} - Couverture: {stats['coverage_rate']:.1%}")
        
        return stats
    
    def _create_fallback_stats(self, stations_df, region_name, error="unknown"):
        """Créer des stats de base en cas d'échec"""
        return {
            'region_name': region_name,
            'total_input_stations': len(stations_df),
            'stations_linked': 0,
            'stations_not_linked': len(stations_df),
            'coverage_rate': 0.0,
            'mean_distance_to_road': np.nan,
            'median_distance_to_road': np.nan,
            'max_distance_to_road': np.nan,
            'unique_routes': 0,
            'routes_with_stations': 0,
            'avg_stations_per_route': 0.0,
            'error': error
        }
    
    def _calculate_quick_stats(self, stations_routes, total_input_stations, region_name):
        """Calcul rapide des statistiques de couverture"""
        
        # Stations liées à des routes (distance < 2km)
        if 'dist_to_highway_m' in stations_routes.columns:
            linked_stations = stations_routes.dropna(subset=['dist_to_highway_m'])
            linked_stations = linked_stations[linked_stations['dist_to_highway_m'] < 2000]
        else:
            print("⚠️ Colonne dist_to_highway_m non trouvée")
            linked_stations = pd.DataFrame()
        
        # Stations uniques (éviter les doublons)
        unique_linked_stations = len(linked_stations.index.unique()) if not linked_stations.empty else 0
        
        stats = {
            'region_name': region_name,
            'total_input_stations': total_input_stations,
            'stations_linked': unique_linked_stations,
            'stations_not_linked': total_input_stations - unique_linked_stations,
            'coverage_rate': unique_linked_stations / total_input_stations if total_input_stations > 0 else 0.0,
        }
        
        if not linked_stations.empty and 'dist_to_highway_m' in linked_stations.columns:
            valid_distances = linked_stations.dropna(subset=['dist_to_highway_m'])
            if not valid_distances.empty:
                stats.update({
                    'mean_distance_to_road': valid_distances['dist_to_highway_m'].mean(),
                    'median_distance_to_road': valid_distances['dist_to_highway_m'].median(),
                    'max_distance_to_road': valid_distances['dist_to_highway_m'].max(),
                })
                
                # Compter les routes avec des stations
                if 'osmid' in valid_distances.columns:
                    routes_with_stations = valid_distances.groupby('osmid').size()
                    stats.update({
                        'unique_routes': len(valid_distances['osmid'].unique()),
                        'routes_with_stations': len(routes_with_stations),
                        'avg_stations_per_route': routes_with_stations.mean(),
                        'max_stations_per_route': routes_with_stations.max(),
                    })
                else:
                    stats.update({
                        'unique_routes': 0,
                        'routes_with_stations': 0,
                        'avg_stations_per_route': 0.0,
                        'max_stations_per_route': 0,
                    })
            else:
                stats.update({
                    'mean_distance_to_road': np.nan,
                    'median_distance_to_road': np.nan,
                    'max_distance_to_road': np.nan,
                    'unique_routes': 0,
                    'routes_with_stations': 0,
                    'avg_stations_per_route': 0.0,
                    'max_stations_per_route': 0,
                })
        else:
            stats.update({
                'mean_distance_to_road': np.nan,
                'median_distance_to_road': np.nan,
                'max_distance_to_road': np.nan,
                'unique_routes': 0,
                'routes_with_stations': 0,
                'avg_stations_per_route': 0.0,
                'max_stations_per_route': 0,
            })
        
        return stats
    
    def analyze_all_regions_quick(self, operator='Orange', regions_subset=None):
        """Analyser rapidement toutes les régions - STATS SEULEMENT"""
        
        # Déterminer quelles régions analyser
        if regions_subset:
            regions_to_analyze = [r for r in regions_subset if r in self.regions]
            print(f"Analyse rapide de {len(regions_to_analyze)} régions spécifiées")
        else:
            regions_to_analyze = self.regions
            print(f"Analyse rapide de toutes les {len(regions_to_analyze)} régions")
        
        results = []
        
        start_time = datetime.now()
        
        for i, region in enumerate(regions_to_analyze, 1):
            print(f"\n{'='*50}")
            print(f"RÉGION {i}/{len(regions_to_analyze)}: {region}")
            print(f"{'='*50}")
            
            region_stats = self.analyze_region_coverage(region, operator)
            if region_stats:
                results.append(region_stats)
                
                # Affichage rapide des résultats
                print(f"   📊 Stations: {region_stats['total_input_stations']}")
                print(f"   🔗 Liées aux routes: {region_stats['stations_linked']}")
                print(f"   📈 Taux couverture: {region_stats['coverage_rate']:.1%}")
                if not np.isnan(region_stats['mean_distance_to_road']):
                    print(f"   📏 Distance moyenne: {region_stats['mean_distance_to_road']:.0f}m")
            else:
                print(f"❌ Échec analyse {region}")
        
        elapsed = datetime.now() - start_time
        print(f"\n⏱️ Temps total: {elapsed}")
        
        # Créer un résumé rapide
        if results:
            self._create_quick_summary(results, operator)
        
        return results
    
    def _create_quick_summary(self, results, operator):
        """Créer un résumé rapide des résultats"""
        
        summary_df = pd.DataFrame(results)
        
        # Calculs globaux
        total_stations = summary_df['total_input_stations'].sum()
        total_linked = summary_df['stations_linked'].sum()
        avg_coverage = summary_df['coverage_rate'].mean()
        
        # Statistiques des distances (ignorer les NaN)
        valid_distances = summary_df.dropna(subset=['mean_distance_to_road'])
        avg_distance = valid_distances['mean_distance_to_road'].mean() if not valid_distances.empty else np.nan
        
        print(f"\n{'='*80}")
        print(f"🇫🇷 RÉSUMÉ RAPIDE - COUVERTURE NATIONALE - OPÉRATEUR {operator}")
        print(f"{'='*80}")
        print(f"📊 Régions analysées: {len(results)}")
        print(f"📡 Total stations: {total_stations:,}")
        print(f"🔗 Stations liées à des routes: {total_linked:,}")
        print(f"📈 Taux de couverture global: {(total_linked/total_stations)*100:.1f}%")
        print(f"📈 Taux de couverture moyen par région: {avg_coverage*100:.1f}%")
        if not np.isnan(avg_distance):
            print(f"📏 Distance moyenne station-route: {avg_distance:.0f}m")
        
        # Top et Bottom régions
        top_coverage = summary_df.nlargest(3, 'coverage_rate')
        bottom_coverage = summary_df.nsmallest(3, 'coverage_rate')
        
        print(f"\n🏆 TOP 3 - Meilleure couverture:")
        for _, row in top_coverage.iterrows():
            print(f"   1. {row['region_name']}: {row['coverage_rate']:.1%} ({row['stations_linked']}/{row['total_input_stations']} stations)")
        
        print(f"\n📉 BOTTOM 3 - Plus faible couverture:")
        for _, row in bottom_coverage.iterrows():
            print(f"   {row['region_name']}: {row['coverage_rate']:.1%} ({row['stations_linked']}/{row['total_input_stations']} stations)")
        
        # Région avec le plus de stations
        max_stations_region = summary_df.loc[summary_df['total_input_stations'].idxmax()]
        print(f"\n📍 Région avec le plus de stations: {max_stations_region['region_name']} ({max_stations_region['total_input_stations']} stations)")
        
        # Export du résumé seulement
        summary_filename = f"{self.output_dir}/RESUME_COUVERTURE_{operator}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with pd.ExcelWriter(summary_filename, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Couverture_par_Region', index=False)
            
            # Feuille résumé global
            global_summary = pd.DataFrame([{
                'Métrique': 'Régions analysées',
                'Valeur': len(results)
            }, {
                'Métrique': 'Total stations',
                'Valeur': total_stations
            }, {
                'Métrique': 'Stations liées',
                'Valeur': total_linked
            }, {
                'Métrique': 'Taux couverture global (%)',
                'Valeur': (total_linked/total_stations)*100
            }, {
                'Métrique': 'Taux couverture moyen (%)',
                'Valeur': avg_coverage*100
            }, {
                'Métrique': 'Distance moyenne (m)',
                'Valeur': avg_distance if not np.isnan(avg_distance) else 'N/A'
            }])
            
            global_summary.to_excel(writer, sheet_name='Résumé_Global', index=False)
        
        print(f"\n📄 Résumé sauvegardé: {summary_filename}")

# ========== UTILISATION RAPIDE ==========

    def diagnose_geopandas_version(self):
        """Diagnostique la version de GeoPandas et les problèmes potentiels"""
        print(f"\n🔍 DIAGNOSTIC GEOPANDAS")
        print(f"Version GeoPandas: {gpd.__version__}")
        
        # Test simple de spatial join
        print("Test spatial join...")
        try:
            # Créer deux GeoDataFrames simples pour tester
            from shapely.geometry import Point, Polygon
            
            # Points de test
            points_data = {
                'id': [1, 2, 3],
                'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
            }
            points_gdf = gpd.GeoDataFrame(points_data, crs="EPSG:4326")
            
            # Polygones de test
            poly_data = {
                'id': [1, 2],
                'geometry': [
                    Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]),
                    Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
                ]
            }
            poly_gdf = gpd.GeoDataFrame(poly_data, crs="EPSG:4326")
            
            # Test du join
            result = gpd.sjoin(points_gdf, poly_gdf, how='left', predicate='intersects')
            
            print(f"✅ Spatial join fonctionne")
            print(f"Colonnes résultat: {list(result.columns)}")
            
            # Identifier la colonne d'index right
            right_cols = [col for col in result.columns if 'right' in col.lower() or 'index' in col.lower()]
            print(f"Colonnes d'index détectées: {right_cols}")
            
        except Exception as e:
            print(f"❌ Erreur test spatial join: {e}")
            print(f"Type d'erreur: {type(e)}")

def main_quick():
    """Fonction principale pour analyse rapide de couverture"""
    
    # Initialiser l'analyseur rapide
    input_file = '2023_T4_sites_Metropole.csv'
    analyzer = RegionalCoverageAnalyzer(input_file)
    
    # Diagnostic GeoPandas
    analyzer.diagnose_geopandas_version()
    
    print(f"\n🚀 ANALYSE RAPIDE DE COUVERTURE")
    print(f"Mode: STATS UNIQUEMENT (pas de cartes ni d'Excel détaillés)")
    print(f"Opérateur: Orange")
    
    # Option 1: Test sur quelques régions d'abord
    print(f"\n❓ Voulez-vous tester sur quelques régions d'abord ? (recommandé)")
    test_response = input("Taper 'test' pour tester sur 3 régions, ou 'all' pour toutes: ")
    
    if test_response.lower() == 'test':
        test_regions = ['Normandie', 'Bretagne', 'Île-de-France']
        print(f"🧪 Test sur {len(test_regions)} régions...")
        results = analyzer.analyze_all_regions_quick(operator='Orange', regions_subset=test_regions)
    else:
        print(f"🚀 Analyse complète de toutes les régions...")
        results = analyzer.analyze_all_regions_quick(operator='Orange')
    
    print(f"\n🎉 Analyse terminée! {len(results)} régions traitées")
    return results

if __name__ == "__main__":
    results = main_quick()