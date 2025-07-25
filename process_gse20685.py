import pandas as pd
import os

file_path = os.path.join("/home", "ubuntu", "GSE20685_series_matrix.txt")

# Read the file, skipping metadata lines starting with '!'
# The actual data starts after a line that typically looks like '!series_matrix_table_begin'
# We need to find where the actual data begins.

with open(file_path, 'r') as f:
    lines = f.readlines()

data_start_line = -1
for i, line in enumerate(lines):
    if line.startswith('!series_matrix_table_begin'):
        data_start_line = i + 1
        break

if data_start_line == -1:
    raise ValueError("Could not find '!series_matrix_table_begin' in the file.")

# Read the data into a pandas DataFrame, skipping the metadata at the beginning
df = pd.read_csv(file_path, sep='\t', skiprows=data_start_line, index_col=0)

# The last line usually contains '!series_matrix_table_end', which needs to be removed
if df.index[-1] == '!series_matrix_table_end':
    df = df.iloc[:-1]

# Transpose the DataFrame so that samples are rows and genes are columns
df = df.T

# Extract clinical labels. These are in lines starting with '!Sample_characteristics_ch1'
clinical_data = {}
sample_geo_accessions = []
sample_titles = []

# First, get the sample GEO accessions and titles
for line in lines:
    if line.startswith('!Sample_geo_accession'):
        sample_geo_accessions = [s.replace('"', '').strip() for s in line.strip().split('\t')[1:]]
    elif line.startswith('!Sample_title'):
        sample_titles = [s.replace('"', '').strip() for s in line.strip().split('\t')[1:]]

if not sample_geo_accessions:
    raise ValueError("Could not find '!Sample_geo_accession' line.")
if not sample_titles:
    raise ValueError("Could not find '!Sample_title' line.")

# Create a mapping from GEO accession to sample title
geo_to_title_map = dict(zip(sample_geo_accessions, sample_titles))

# Now process the characteristics lines
for line in lines:
    if line.startswith('!Sample_characteristics_ch1'):
        parts = line.strip().split('\t')
        
        # The characteristic name is usually the first part, cleaned up
        characteristic_name_raw = parts[0].replace('!Sample_characteristics_ch1', '').strip()
        
        values = [p.replace('"', '').strip() for p in parts[1:]]
        
        if len(sample_geo_accessions) == len(values):
            for i, geo_accession in enumerate(sample_geo_accessions):
                sample_title = geo_to_title_map.get(geo_accession, geo_accession) # Use GEO accession if title not found
                if sample_title not in clinical_data:
                    clinical_data[sample_title] = {}
                
                # If the characteristic name is empty, it means the actual characteristic is within the value itself
                if ':' in values[i] and not characteristic_name_raw:
                    key, val = values[i].split(':', 1)
                    clinical_data[sample_title][key.strip()] = val.strip()
                elif characteristic_name_raw:
                    clinical_data[sample_title][characteristic_name_raw] = values[i]
                else:
                    # Fallback for unexpected formats, assign a generic name or skip
                    print(f"Warning: Could not parse characteristic for sample {sample_title} from value: {values[i]}")
        else:
            print(f"Warning: Mismatch in sample GEO accessions ({len(sample_geo_accessions)}) and characteristic values ({len(values)}) for line: {line}")

# Convert clinical_data dictionary to a DataFrame
clinical_df = pd.DataFrame.from_dict(clinical_data, orient='index')

# Clean up column names in clinical_df: remove empty strings and handle duplicates
# Create a list of new column names, ensuring uniqueness
new_columns = []
seen_columns = set()
for col in clinical_df.columns:
    if col and col not in seen_columns:
        new_columns.append(col)
        seen_columns.add(col)
    elif col:
        # Handle duplicate column names by appending a suffix
        suffix = 1
        while f"{col}_{suffix}" in seen_columns:
            suffix += 1
        new_columns.append(f"{col}_{suffix}")
        seen_columns.add(f"{col}_{suffix}")
    else:
        # Handle empty column names, assign a generic name
        suffix = 1
        while f"unnamed_col_{suffix}" in seen_columns:
            suffix += 1
        new_columns.append(f"unnamed_col_{suffix}")
        seen_columns.add(f"unnamed_col_{suffix}")

clinical_df.columns = new_columns

# Rename the index of df to match the sample titles for merging
df.index = df.index.map(geo_to_title_map)

# Print indices before merging
print("\nIndex of gene expression data (df):")
print(df.index[:10])  # Show first 10 for brevity
print("\nIndex of clinical data (clinical_df):")
print(clinical_df.index[:10])  # Show first 10 for brevity

# Merge gene expression data and clinical data
# Ensure that the indices (sample IDs) match
merged_df = pd.merge(df, clinical_df, left_index=True, right_index=True, how='inner')

# Save the processed data and clinical data for later use
merged_df.to_csv('/home/ubuntu/processed_gse20685_data.csv')
clinical_df.to_csv('/home/ubuntu/gse20685_clinical_data.csv')

print("Data processing complete. Merged data saved to /home/ubuntu/processed_gse20685_data.csv")
print("Clinical data saved to /home/ubuntu/gse20685_clinical_data.csv")
print(f"Shape of gene expression data: {df.shape}")
print(f"Shape of clinical data: {clinical_df.shape}")
print(f"Shape of merged data: {merged_df.shape}")

# Display first few rows of the merged data and clinical data
print("\nFirst 5 rows of merged data:")
print(merged_df.head())
print("\nFirst 5 rows of clinical data:")
print(clinical_df.head())

# Check for missing values in the merged data
print("\nMissing values in merged data:")
print(merged_df.isnull().sum().sum())

# Check for unique values in some clinical characteristics to understand their distribution
print("\nUnique values for 'subtype':")
if 'subtype' in clinical_df.columns:
    print(clinical_df['subtype'].value_counts())
else:
    print("'subtype' column not found in clinical data.")

print("\nUnique values for 'event_death':")
if 'event_death' in clinical_df.columns:
    print(clinical_df['event_death'].value_counts())
else:
    print("'event_death' column not found in clinical data.")

print("\nUnique values for 'event_metastasis':")
if 'event_metastasis' in clinical_df.columns:
    print(clinical_df['event_metastasis'].value_counts())
else:
    print("'event_metastasis' column not found in clinical data.")

print("\nUnique values for 'adjuvant_chemotherapy':")
if 'adjuvant_chemotherapy' in clinical_df.columns:
    print(clinical_df['adjuvant_chemotherapy'].value_counts())
else:
    print("'adjuvant_chemotherapy' column not found in clinical data.")



