import pandas as pd

# Chemins des fichiers
input_file = '2023_T4_sites_Metropole.csv'
output_file = 'stations_normandie_orange.csv'

# Lecture du CSV original avec point-virgule comme séparateur
df = pd.read_csv(input_file, sep=';')

# Filtrage : uniquement la région "Normandie" et l'opérateur "Orange"
filtered = df[(df['nom_reg'] == "Normandie") & (df['nom_op'] == 'Orange')]

# Enregistrement du nouveau CSV
filtered.to_csv(output_file, index=False, sep=';')


