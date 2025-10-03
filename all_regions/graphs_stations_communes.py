import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class TelecomAnalyzer:
    def __init__(self, csv_file_path):
        """
        Initialise l'analyseur avec le fichier CSV des sites télécom.
        
        Args:
            csv_file_path (str): Chemin vers le fichier CSV
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.common_stations = None
        self.exact_match = True  # Comparaison exacte des coordonnées
        
    def load_data(self):
        """
        Charge et nettoie les données du fichier CSV délimité par des points-virgules.
        """
        print("📂 Chargement des données (format point-virgule)...")
        
        try:
            # Vérifier si le fichier existe
            import os
            if not os.path.exists(self.csv_file_path):
                print(f"❌ Fichier non trouvé: {self.csv_file_path}")
                return False
            
            print(f"📄 Fichier trouvé: {self.csv_file_path}")
            file_size = os.path.getsize(self.csv_file_path) / (1024*1024)
            print(f"📏 Taille: {file_size:.1f} MB")
            
            # Charger avec point-virgule comme séparateur
            print("🔄 Chargement avec délimiteur ';'...")
            
            # Essayer différents encodages pour les fichiers français
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    print(f"   Essai avec encoding: {encoding}")
                    self.data = pd.read_csv(
                        self.csv_file_path, 
                        sep=';',  # Point-virgule explicite
                        encoding=encoding,
                        low_memory=False,
                        on_bad_lines='skip'  # Ignorer les lignes malformées
                    )
                    
                    print(f"   ✅ Chargé avec {encoding}: {len(self.data)} lignes")
                    break
                    
                except UnicodeDecodeError:
                    print(f"   ❌ Échec encoding {encoding}")
                    continue
                except Exception as e:
                    print(f"   ❌ Erreur avec {encoding}: {e}")
                    continue
            else:
                print("❌ Tous les encodages ont échoué")
                return False
            
            print(f"📊 Lignes brutes chargées: {len(self.data):,}")
            
            # Nettoyer les noms de colonnes
            print("🧹 Nettoyage des colonnes...")
            original_columns = list(self.data.columns)
            self.data.columns = [col.strip() for col in self.data.columns]
            
            print(f"📋 Colonnes disponibles: {list(self.data.columns)}")
            
            # Vérifier les colonnes nécessaires
            required_columns = ['latitude', 'longitude', 'nom_op', 'nom_dep']
            available_columns = list(self.data.columns)
            
            print(f"🔍 Vérification des colonnes requises:")
            missing_columns = []
            for col in required_columns:
                if col in available_columns:
                    print(f"   ✅ {col}: trouvée")
                else:
                    print(f"   ❌ {col}: MANQUANTE")
                    # Chercher des colonnes similaires
                    similar = [c for c in available_columns if col.lower() in c.lower() or c.lower() in col.lower()]
                    if similar:
                        print(f"      Colonnes similaires: {similar}")
                    missing_columns.append(col)
            
            if missing_columns:
                print(f"❌ Colonnes manquantes critiques: {missing_columns}")
                print(f"💡 Colonnes disponibles dans le fichier:")
                for i, col in enumerate(available_columns, 1):
                    print(f"   {i:2d}. {col}")
                return False
            
            # Nettoyer les données étape par étape
            print(f"\n🧹 Nettoyage des données:")
            initial_count = len(self.data)
            print(f"   Départ: {initial_count:,} lignes")
            
            # Supprimer les lignes vides dans les colonnes critiques
            for col in required_columns:
                before = len(self.data)
                # Supprimer les NaN et les chaînes vides
                self.data = self.data[
                    self.data[col].notna() & 
                    (self.data[col] != '') & 
                    (self.data[col] != ' ')
                ]
                after = len(self.data)
                if before != after:
                    print(f"   Après nettoyage '{col}': {after:,} lignes (-{before-after:,})")
            
            if len(self.data) == 0:
                print("❌ Aucune ligne valide après nettoyage des valeurs manquantes")
                return False
            
            # Convertir les coordonnées en numérique
            print("🔢 Conversion des coordonnées GPS...")
            
            # Remplacer les virgules par des points si nécessaire (format français)
            if self.data['latitude'].dtype == 'object':
                self.data['latitude'] = self.data['latitude'].astype(str).str.replace(',', '.')
            if self.data['longitude'].dtype == 'object':
                self.data['longitude'] = self.data['longitude'].astype(str).str.replace(',', '.')
            
            self.data['latitude'] = pd.to_numeric(self.data['latitude'], errors='coerce')
            self.data['longitude'] = pd.to_numeric(self.data['longitude'], errors='coerce')
            
            # Vérifier les conversions
            lat_invalid = self.data['latitude'].isna().sum()
            lon_invalid = self.data['longitude'].isna().sum()
            if lat_invalid > 0:
                print(f"   ⚠️ {lat_invalid} latitudes invalides détectées")
            if lon_invalid > 0:
                print(f"   ⚠️ {lon_invalid} longitudes invalides détectées")
            
            # Supprimer les coordonnées invalides
            before = len(self.data)
            self.data = self.data.dropna(subset=['latitude', 'longitude'])
            after = len(self.data)
            if before != after:
                print(f"   Après suppression coordonnées invalides: {after:,} lignes (-{before-after:,})")
            
            # Filtrer les coordonnées dans la plage de la France métropolitaine
            before = len(self.data)
            france_filter = (
                (self.data['latitude'].between(41.0, 51.5)) &
                (self.data['longitude'].between(-5.5, 9.5))
            )
            self.data = self.data[france_filter]
            after = len(self.data)
            if before != after:
                print(f"   Après filtrage géographique France: {after:,} lignes (-{before-after:,})")
            
            if len(self.data) == 0:
                print("❌ Aucune donnée valide après filtrage géographique")
                return False
            
            # Statistiques finales
            print(f"\n✅ Données chargées avec succès:")
            print(f"   📊 Sites totaux: {len(self.data):,}")
            print(f"   🏢 Opérateurs uniques: {self.data['nom_op'].nunique()}")
            print(f"   🗺️ Départements uniques: {self.data['nom_dep'].nunique()}")
            
            # Afficher les opérateurs
            operators = self.data['nom_op'].unique()
            print(f"   📡 Opérateurs: {', '.join(operators)}")
            
            # Afficher quelques départements
            departments = self.data['nom_dep'].unique()[:10]
            print(f"   🏛️ Premiers départements: {', '.join(departments)}")
            
            # Échantillon des données
            print(f"\n📋 Échantillon des données:")
            sample = self.data[['nom_op', 'nom_dep', 'latitude', 'longitude']].head(3)
            for idx, row in sample.iterrows():
                print(f"   {row['nom_op']} | {row['nom_dep']} | {row['latitude']:.5f}, {row['longitude']:.5f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def find_common_stations(self):
        """
        Identifie les stations communes par comparaison EXACTE des coordonnées GPS.
        Une station commune = mêmes coordonnées latitude/longitude exactes.
        """
        print("🔍 Recherche des stations communes (coordonnées exactes)...")
        
        # Créer une clé unique basée sur les coordonnées exactes
        self.data['coord_key'] = self.data['latitude'].astype(str) + '_' + self.data['longitude'].astype(str)
        
        print(f"📊 Coordonnées uniques trouvées: {self.data['coord_key'].nunique():,}")
        
        # Grouper par coordonnées exactes
        location_groups = self.data.groupby('coord_key')
        
        common_stations_list = []
        multiple_sites_same_coords = 0
        multiple_operators_same_coords = 0
        
        for coord_key, group in location_groups:
            if len(group) > 1:  # Plusieurs sites aux mêmes coordonnées
                multiple_sites_same_coords += 1
                operators = group['nom_op'].unique()
                
                if len(operators) > 1:  # Plusieurs opérateurs différents
                    multiple_operators_same_coords += 1
                    
                    station_info = {
                        'coord_key': coord_key,
                        'latitude': group['latitude'].iloc[0],  # Coordonnées exactes
                        'longitude': group['longitude'].iloc[0],
                        'operators': list(operators),
                        'operator_count': len(operators),
                        'site_count': len(group),
                        'department': group['nom_dep'].iloc[0],
                        'region': group['nom_reg'].iloc[0] if 'nom_reg' in group.columns else 'N/A',
                        'commune': group['nom_com'].iloc[0] if 'nom_com' in group.columns else 'N/A',
                        'sites_details': group[['nom_op', 'num_site']].to_dict('records') if 'num_site' in group.columns else []
                    }
                    common_stations_list.append(station_info)
        
        self.common_stations = pd.DataFrame(common_stations_list)
        
        print(f"📍 Résultats de l'analyse:")
        print(f"   • {multiple_sites_same_coords:,} coordonnées avec plusieurs sites")
        print(f"   • {multiple_operators_same_coords:,} coordonnées avec plusieurs opérateurs")
        print(f"   • {len(self.common_stations):,} stations communes identifiées")
        
        if len(self.common_stations) > 0:
            # Statistiques sur le partage
            max_operators = self.common_stations['operator_count'].max()
            avg_operators = self.common_stations['operator_count'].mean()
            print(f"   • Maximum d'opérateurs sur une station: {max_operators}")
            print(f"   • Moyenne d'opérateurs par station commune: {avg_operators:.1f}")
        
        return self.common_stations
    
    def analyze_by_operator(self):
        """
        Analyse les statistiques par opérateur.
        """
        if self.common_stations is None:
            self.find_common_stations()
        
        print("\n📈 Analyse par opérateur:")
        
        # Compter les occurrences de chaque opérateur dans les stations communes
        operator_counts = defaultdict(int)
        
        for _, station in self.common_stations.iterrows():
            for operator in station['operators']:
                operator_counts[operator] += 1
        
        operator_stats = pd.DataFrame(list(operator_counts.items()), 
                                    columns=['Opérateur', 'Stations_Communes'])
        operator_stats = operator_stats.sort_values('Stations_Communes', ascending=False)
        
        print(operator_stats.to_string(index=False))
        
        return operator_stats
    
    def analyze_by_department(self):
        """
        Analyse les statistiques par département.
        """
        if self.common_stations is None:
            self.find_common_stations()
        
        print("\n🗺️ Analyse par département:")
        
        dept_stats = self.common_stations['department'].value_counts().head(15)
        dept_df = pd.DataFrame({
            'Département': dept_stats.index,
            'Stations_Communes': dept_stats.values,
            'Pourcentage': (dept_stats.values / len(self.common_stations) * 100).round(1)
        })
        
        print(dept_df.to_string(index=False))
        
        return dept_df
    
    def create_operator_sharing_matrix(self):
        """
        Crée une matrice montrant combien de stations communes chaque paire d'opérateurs partage.
        """
        if self.common_stations is None or len(self.common_stations) == 0:
            print("❌ Aucune station commune pour créer la matrice")
            return None
        
        print("📊 Création de la matrice de partage entre opérateurs...")
        
        # Obtenir tous les opérateurs uniques
        all_operators = sorted(list(set([op for station in self.common_stations['operators'] for op in station])))
        
        print(f"   Opérateurs détectés: {all_operators}")
        
        # Créer une matrice vide
        matrix = pd.DataFrame(0, index=all_operators, columns=all_operators)
        
        # Remplir la matrice
        for _, station in self.common_stations.iterrows():
            operators = station['operators']
            # Pour chaque paire d'opérateurs dans cette station
            for i, op1 in enumerate(operators):
                for j, op2 in enumerate(operators):
                    if i != j:  # Éviter de compter un opérateur avec lui-même
                        matrix.loc[op1, op2] += 1
        
        # La matrice est symétrique, donc diviser par 2 pour éviter le double comptage
        # Mais garder la diagonale à 0
        for i in range(len(all_operators)):
            for j in range(i+1, len(all_operators)):
                op1, op2 = all_operators[i], all_operators[j]
                shared_count = matrix.loc[op1, op2]
                matrix.loc[op1, op2] = shared_count
                matrix.loc[op2, op1] = shared_count
        
        print(f"✅ Matrice de partage créée ({len(all_operators)}x{len(all_operators)})")
        
        # Afficher la matrice
        print(f"\n📋 MATRICE DE PARTAGE ENTRE OPÉRATEURS:")
        print(f"   (Nombre de stations communes par paire d'opérateurs)")
        print(matrix)
        
        # Statistiques intéressantes
        print(f"\n🎯 STATISTIQUES DE PARTAGE:")
        
        # Paires les plus partagées
        upper_triangle = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
        max_sharing = upper_triangle.max().max()
        
        if max_sharing > 0:
            max_pairs = []
            for col in upper_triangle.columns:
                for idx in upper_triangle.index:
                    if upper_triangle.loc[idx, col] == max_sharing:
                        max_pairs.append((idx, col, max_sharing))
            
            print(f"   🏆 Paire(s) avec le plus de partage ({max_sharing} stations):")
            for op1, op2, count in max_pairs:
                print(f"      • {op1} ↔ {op2}: {count} stations communes")
        
        # Total de partages par opérateur
        sharing_totals = matrix.sum(axis=1) / 2  # Diviser par 2 car matrice symétrique
        sharing_totals = sharing_totals.sort_values(ascending=False)
        
        print(f"\n   📊 Total de partages par opérateur:")
        for op, total in sharing_totals.items():
            print(f"      • {op}: {int(total)} partages")
        
        return matrix
    
    def visualize_sharing_matrix(self, matrix=None):
        """
        Crée une heatmap de la matrice de partage.
        """
        if matrix is None:
            matrix = self.create_operator_sharing_matrix()
        
        if matrix is None:
            return
        
        try:
            # Essayer avec seaborn (plus joli)
            plt.figure(figsize=(10, 8))
            
            sns.heatmap(matrix, 
                       annot=True,  # Afficher les valeurs
                       fmt='d',     # Format entier
                       cmap='Blues', 
                       square=True,
                       cbar_kws={'label': 'Nombre de stations communes'},
                       linewidths=0.5)
            
            plt.title('Matrice de Partage entre Opérateurs\n(Nombre de stations communes)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Opérateurs', fontweight='bold')
            plt.ylabel('Opérateurs', fontweight='bold')
            
            # Rotation des labels pour une meilleure lisibilité
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("⚠️ Seaborn non disponible, création d'une heatmap simple...")
            self._create_simple_matrix_plot(matrix)
        except Exception as e:
            print(f"⚠️ Erreur avec seaborn: {e}")
            self._create_simple_matrix_plot(matrix)
    
    def _create_simple_matrix_plot(self, matrix):
        """
        Crée une heatmap simple avec matplotlib seulement.
        """
        plt.figure(figsize=(10, 8))
        
        # Créer la heatmap avec matplotlib
        im = plt.imshow(matrix.values, cmap='Blues', aspect='auto')
        
        # Ajouter les valeurs sur chaque case
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                plt.text(j, i, int(matrix.iloc[i, j]), 
                        ha='center', va='center', fontweight='bold')
        
        # Configuration des axes
        plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(matrix.index)), matrix.index)
        
        plt.title('Matrice de Partage entre Opérateurs\n(Nombre de stations communes)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Opérateurs', fontweight='bold')
        plt.ylabel('Opérateurs', fontweight='bold')
        
        # Ajouter une barre de couleur
        cbar = plt.colorbar(im)
        cbar.set_label('Nombre de stations communes', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()
    
    def print_matrix_table(self, matrix=None):
        """
        Affiche la matrice sous forme de tableau bien formaté dans la console.
        """
        if matrix is None:
            matrix = self.create_operator_sharing_matrix()
        
        if matrix is None:
            return
        
        print("\n" + "="*80)
        print("📊 MATRICE DE PARTAGE ENTRE OPÉRATEURS")
        print("="*80)
        print("(Nombre de stations communes par paire d'opérateurs)")
        print()
        
        # Formatter pour un affichage propre
        operators = list(matrix.columns)
        
        # En-tête
        header = "           "
        for op in operators:
            header += f"{op:>8s} "
        print(header)
        print("-" * len(header))
        
        # Lignes de données
        for i, op1 in enumerate(operators):
            row = f"{op1:>10s} "
            for j, op2 in enumerate(operators):
                value = int(matrix.loc[op1, op2])
                if i == j:  # Diagonale
                    row += f"{'':>8s} "
                else:
                    row += f"{value:>8d} "
            print(row)
        
        print("\n💡 Lecture: La case (Ligne, Colonne) indique combien de stations")
        print("   les deux opérateurs partagent ensemble.")
        print("="*80)
    
    def analyze_operator_pairs(self):
        """
        Analyse détaillée des paires d'opérateurs qui partagent le plus.
        """
        if self.common_stations is None or len(self.common_stations) == 0:
            print("❌ Aucune donnée pour l'analyse des paires")
            return None
        
        print("🔍 Analyse détaillée des paires d'opérateurs...")
        
        # Compter les occurrences de chaque paire
        pair_counts = {}
        pair_stations = {}  # Pour stocker les détails des stations
        
        for idx, station in self.common_stations.iterrows():
            operators = sorted(station['operators'])  # Tri pour cohérence
            station_info = {
                'commune': station['commune'],
                'department': station['department'],
                'latitude': station['latitude'],
                'longitude': station['longitude']
            }
            
            # Générer toutes les paires possibles
            from itertools import combinations
            for pair in combinations(operators, 2):
                pair_key = f"{pair[0]} + {pair[1]}"
                
                if pair_key not in pair_counts:
                    pair_counts[pair_key] = 0
                    pair_stations[pair_key] = []
                
                pair_counts[pair_key] += 1
                pair_stations[pair_key].append(station_info)
        
        # Convertir en DataFrame et trier
        pairs_df = pd.DataFrame(list(pair_counts.items()), 
                               columns=['Paire_Opérateurs', 'Stations_Communes'])
        pairs_df = pairs_df.sort_values('Stations_Communes', ascending=False)
        
        print(f"\n📊 TOP 10 DES PAIRES D'OPÉRATEURS:")
        for idx, row in pairs_df.head(10).iterrows():
            pair_name = row['Paire_Opérateurs']
            count = row['Stations_Communes']
            print(f"   {idx+1:2d}. {pair_name}: {count} stations communes")
            
            # Afficher quelques exemples de stations
            if count <= 3:  # Si peu de stations, afficher toutes
                stations = pair_stations[pair_name]
                for station in stations:
                    print(f"       📍 {station['commune']} ({station['department']})")
            else:  # Sinon, afficher les 3 premières
                stations = pair_stations[pair_name][:3]
                for station in stations:
                    print(f"       📍 {station['commune']} ({station['department']})")
                if count > 3:
                    print(f"       ... et {count-3} autres stations")
            print()
        
        return pairs_df, pair_stations
        """
        Analyse alternative utilisant la colonne id_site_partage si disponible.
        """
        if 'id_site_partage' not in self.data.columns:
            print("❌ Colonne 'id_site_partage' non disponible")
            return None
        
        print("🔍 Analyse par id_site_partage...")
        
        # Filtrer les sites avec un ID de partage
        shared_sites = self.data[self.data['id_site_partage'].notna() & (self.data['id_site_partage'] != '')]
        
        if len(shared_sites) == 0:
            print("❌ Aucun site avec id_site_partage renseigné")
            return None
        
        print(f"📊 {len(shared_sites)} sites avec id_site_partage")
        
        # Grouper par ID de partage
        sharing_groups = shared_sites.groupby('id_site_partage')
        
        shared_infrastructure = []
        
        for share_id, group in sharing_groups:
            if len(group) > 1:
                operators = group['nom_op'].unique()
                if len(operators) > 1:
                    shared_info = {
                        'share_id': share_id,
                        'operators': list(operators),
                        'operator_count': len(operators),
                        'site_count': len(group),
                        'departments': list(group['nom_dep'].unique()),
                        'coordinates': list(group[['latitude', 'longitude']].drop_duplicates().values)
                    }
                    shared_infrastructure.append(shared_info)
        
        print(f"🎯 {len(shared_infrastructure)} infrastructures partagées via id_site_partage")
        
        return pd.DataFrame(shared_infrastructure)
    def get_top_shared_stations(self, n=20):
        """
        Retourne les stations avec le plus d'opérateurs.
        """
        if self.common_stations is None or len(self.common_stations) == 0:
            print("❌ Aucune station commune à afficher")
            return pd.DataFrame()
        
        top_stations = self.common_stations.nlargest(n, 'operator_count')[
            ['commune', 'department', 'operator_count', 'operators', 'latitude', 'longitude', 'site_count']
        ]
        
        print(f"\n🏆 Top {min(n, len(top_stations))} des stations avec le plus d'opérateurs (coordonnées exactes):")
        for idx, station in top_stations.iterrows():
            operators_str = ', '.join(station['operators'])
            print(f"   📍 {station['commune']} ({station['department']})")
            print(f"      🏢 {station['operator_count']} opérateurs: {operators_str}")
            print(f"      📊 {station['site_count']} sites | 🗺️ {station['latitude']:.6f}, {station['longitude']:.6f}")
            print()
        
        return top_stations
    
    def create_visualizations(self):
        """
        Crée les graphiques d'analyse.
        """
        if self.common_stations is None:
            self.find_common_stations()
        
        # Configuration des graphiques
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 Analyse des Stations Télécom Communes', fontsize=16, fontweight='bold')
        
        # 1. Stations communes par opérateur
        operator_stats = self.analyze_by_operator()
        axes[0, 0].bar(operator_stats['Opérateur'], operator_stats['Stations_Communes'], 
                       color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Stations communes par opérateur')
        axes[0, 0].set_xlabel('Opérateur')
        axes[0, 0].set_ylabel('Nombre de stations communes')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Top 10 départements
        dept_stats = self.analyze_by_department().head(10)
        axes[0, 1].barh(dept_stats['Département'], dept_stats['Stations_Communes'], 
                        color='lightcoral')
        axes[0, 1].set_title('Top 10 départements - Stations communes')
        axes[0, 1].set_xlabel('Nombre de stations communes')
        
        # 3. Distribution du nombre d'opérateurs par station
        operator_count_dist = self.common_stations['operator_count'].value_counts().sort_index()
        axes[1, 0].bar(operator_count_dist.index, operator_count_dist.values, 
                       color='lightgreen', edgecolor='darkgreen')
        axes[1, 0].set_title('Distribution: Nombre d\'opérateurs par station')
        axes[1, 0].set_xlabel('Nombre d\'opérateurs')
        axes[1, 0].set_ylabel('Nombre de stations')
        
        # 4. Répartition géographique (sample)
        sample_stations = self.common_stations.sample(min(1000, len(self.common_stations)))
        scatter = axes[1, 1].scatter(sample_stations['longitude'], sample_stations['latitude'], 
                                   c=sample_stations['operator_count'], cmap='viridis', 
                                   alpha=0.6, s=20)
        axes[1, 1].set_title('Répartition géographique (échantillon)')
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        plt.colorbar(scatter, ax=axes[1, 1], label='Nb opérateurs')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filename='stations_communes_resultats.csv'):
        """
        Exporte les résultats vers un fichier CSV.
        """
        if self.common_stations is None:
            self.find_common_stations()
        
        # Préparer les données pour l'export
        export_data = self.common_stations.copy()
        export_data['operators_list'] = export_data['operators'].apply(lambda x: ' | '.join(x))
        
        export_columns = ['commune', 'department', 'region', 'latitude', 'longitude', 
                         'operator_count', 'operators_list', 'site_count']
        
        export_data[export_columns].to_csv(filename, index=False, sep=';', encoding='utf-8')
        print(f"💾 Résultats exportés vers {filename}")
    
    def generate_summary_report(self):
        """
        Génère un rapport de synthèse complet.
        """
        if self.data is None or len(self.data) == 0:
            print("❌ Aucune donnée disponible pour générer le rapport")
            return
            
        if self.common_stations is None:
            self.find_common_stations()
        
        print("\n" + "="*60)
        print("📋 RAPPORT DE SYNTHÈSE - STATIONS TÉLÉCOM COMMUNES")
        print("="*60)
        
        print(f"\n📊 STATISTIQUES GÉNÉRALES:")
        print(f"   • Sites totaux analysés: {len(self.data):,}")
        print(f"   • Stations communes identifiées: {len(self.common_stations):,}")
        
        if len(self.data) > 0:
            print(f"   • Pourcentage de mutualisation: {len(self.common_stations)/len(self.data)*100:.1f}%")
        else:
            print(f"   • Pourcentage de mutualisation: 0.0%")
            
        print(f"   • Opérateurs uniques: {self.data['nom_op'].nunique()}")
        
        if len(self.common_stations) > 0:
            print(f"   • Départements concernés: {self.common_stations['department'].nunique()}")
            
            print(f"\n🎯 ANALYSE DES PARTAGES:")
            operator_count_dist = self.common_stations['operator_count'].value_counts()
            for count, freq in operator_count_dist.items():
                print(f"   • {freq} stations partagées par {count} opérateurs")
            
            print(f"\n🏆 TOP 5 OPÉRATEURS (stations communes):")
            operator_stats = self.analyze_by_operator().head(5)
            for _, row in operator_stats.iterrows():
                print(f"   • {row['Opérateur']}: {row['Stations_Communes']} stations")
            
            print(f"\n🗺️ TOP 5 DÉPARTEMENTS:")
            dept_stats = self.analyze_by_department().head(5)
            for _, row in dept_stats.iterrows():
                print(f"   • {row['Département']}: {row['Stations_Communes']} stations ({row['Pourcentage']}%)")
        else:
            print(f"   • Aucune station commune détectée avec coordonnées exactes")
            
            # Diagnostics supplémentaires
            print(f"\n🔍 DIAGNOSTICS:")
            print(f"   • Opérateurs uniques dans les données: {list(self.data['nom_op'].unique())}")
            print(f"   • Sites totaux: {len(self.data):,}")
            print(f"   • Coordonnées uniques: {self.data['coord_key'].nunique():,}")
            
            # Vérifier s'il y a des coordonnées dupliquées (même si même opérateur)
            coord_counts = self.data['coord_key'].value_counts()
            duplicated_coords = coord_counts[coord_counts > 1]
            
            if len(duplicated_coords) > 0:
                print(f"   • {len(duplicated_coords)} coordonnées avec plusieurs sites (même opérateur ou différents)")
                print(f"   • Exemple de coordonnées dupliquées:")
                for coord, count in duplicated_coords.head(3).items():
                    sample_sites = self.data[self.data['coord_key'] == coord]
                    operators_at_coord = sample_sites['nom_op'].unique()
                    lat, lon = coord.split('_')
                    print(f"     - {lat}, {lon}: {count} sites, opérateurs: {list(operators_at_coord)}")
            else:
                print(f"   • Aucune coordonnée dupliquée trouvée - chaque site a des coordonnées uniques")
                print(f"   • Cela signifie qu'il n'y a pas de partage d'infrastructure détectable")
            
            # Suggestion d'analyse alternative
            print(f"\n💡 SUGGESTION:")  
            print(f"   • Vérifiez si le fichier contient une colonne 'id_site_partage' ou similaire")
            print(f"   • Cette colonne pourrait indiquer les sites partagés même avec coordonnées différentes")



def main():
    """
    Fonction principale pour exécuter l'analyse complète.
    """
    # Chemin vers votre fichier CSV - MODIFIEZ CE CHEMIN
    csv_file = "2023_T4_sites_Metropole.csv"
    
    print("🗼 ANALYSEUR DE STATIONS TÉLÉCOM COMMUNES")
    print("="*50)
    
    # Créer l'analyseur
    analyzer = TelecomAnalyzer(csv_file)
    
    # Charger les données avec diagnostic détaillé
    if not analyzer.load_data():
        print("\n❌ Échec du chargement des données.")
        print("🔧 Solutions possibles:")
        print("   1. Vérifiez le chemin du fichier CSV")
        print("   2. Vérifiez l'encodage du fichier (UTF-8, Latin-1, etc.)")
        print("   3. Vérifiez le séparateur (';' ou ',')")
        print("   4. Vérifiez que les colonnes requises existent:")
        print("      - latitude, longitude, nom_op, nom_dep")
        return
    
    # Effectuer l'analyse complète
    analyzer.find_common_stations()
    analyzer.generate_summary_report()
    
    # Si des stations communes sont trouvées, continuer l'analyse
    if len(analyzer.common_stations) > 0:
        # Analyses détaillées
        print("\n" + "="*60)
        analyzer.analyze_by_operator()
        analyzer.analyze_by_department()
        analyzer.get_top_shared_stations()
        
        # Créer les visualisations
        try:
            print("\n📈 Génération des graphiques...")
            analyzer.create_visualizations()
        except Exception as e:
            print(f"⚠️ Erreur lors de la création des graphiques: {e}")
        
        # 4. Matrice de partage entre opérateurs
        sharing_matrix = analyzer.create_operator_sharing_matrix()
        if sharing_matrix is not None:
            # Afficher la matrice dans la console
            analyzer.print_matrix_table(sharing_matrix)
            # Créer le graphique
            analyzer.visualize_sharing_matrix(sharing_matrix)
        
        # 5. Analyse des paires d'opérateurs
        analyzer.analyze_operator_pairs()
        
        # Exporter les résultats
        try:
            analyzer.export_results()
        except Exception as e:
            print(f"⚠️ Erreur lors de l'export: {e}")
    else:
        print("\n💡 SUGGESTIONS POUR TROUVER DES STATIONS COMMUNES:")
        print("   1. Augmentez la tolérance: analyzer.tolerance = 0.001")
        print("   2. Vérifiez que plusieurs opérateurs sont présents")
        print("   3. Examinez la qualité des coordonnées GPS")
    
    print("\n✅ Analyse terminée!")


# FONCTION DE DIAGNOSTIC SUPPLÉMENTAIRE
def diagnose_csv(csv_file):
    """
    Fonction utilitaire pour diagnostiquer un fichier CSV problématique.
    """
    import os
    
    print(f"🔍 DIAGNOSTIC DU FICHIER: {csv_file}")
    print("="*50)
    
    if not os.path.exists(csv_file):
        print(f"❌ Fichier non trouvé: {csv_file}")
        return
    
    # Informations sur le fichier
    file_size = os.path.getsize(csv_file) / (1024*1024)  # MB
    print(f"📄 Taille du fichier: {file_size:.1f} MB")
    
    # Lire les premières lignes
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = [f.readline().strip() for _ in range(5)]
        print(f"📋 Premières lignes (UTF-8):")
        for i, line in enumerate(lines):
            print(f"   {i+1}: {line[:100]}...")
    except:
        try:
            with open(csv_file, 'r', encoding='latin-1') as f:
                lines = [f.readline().strip() for _ in range(5)]
            print(f"📋 Premières lignes (Latin-1):")
            for i, line in enumerate(lines):
                print(f"   {i+1}: {line[:100]}...")
        except Exception as e:
            print(f"❌ Impossible de lire le fichier: {e}")
    
    # Détecter le séparateur probable
    if lines:
        separators = [';', ',', '\t', '|']
        separator_counts = {sep: lines[0].count(sep) for sep in separators}
        likely_sep = max(separator_counts, key=separator_counts.get)
        print(f"🔍 Séparateur probable: '{likely_sep}' ({separator_counts[likely_sep]} occurrences)")
        
        columns = lines[0].split(likely_sep)
        print(f"📊 Colonnes détectées ({len(columns)}): {columns[:5]}...")


# Décommentez cette ligne pour diagnostiquer votre fichier
# diagnose_csv("2023_T4_sites_Metropole.csv")


if __name__ == "__main__":
    main()


# EXEMPLE DE MATRICE DE PARTAGE RÉSULTANTE:
"""
📋 MATRICE DE PARTAGE ENTRE OPÉRATEURS:
(Nombre de stations communes par paire d'opérateurs)

           Orange  SFR  Free  Bouygues
Orange         0   45    23        18
SFR           45    0    31        22  
Free          23   31     0        15
Bouygues      18   22    15         0

🎯 STATISTIQUES DE PARTAGE:
🏆 Paire avec le plus de partage (45 stations):
   • Orange ↔ SFR: 45 stations communes

📊 Total de partages par opérateur:
   • Orange: 86 partages
   • SFR: 98 partages  
   • Free: 69 partages
   • Bouygues: 55 partages

📊 TOP 10 DES PAIRES D'OPÉRATEURS:
 1. Orange + SFR: 45 stations communes
    📍 Paris 15e (Paris)
    📍 Lyon 3e (Rhône)
    📍 Marseille 8e (Bouches-du-Rhône)
    ... et 42 autres stations

 2. Free + SFR: 31 stations communes
    📍 Toulouse 1er (Haute-Garonne)
    📍 Nice (Alpes-Maritimes)
    ... et 29 autres stations
"""