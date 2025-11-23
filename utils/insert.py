"""
This module is used to insert data into the Supabase database.
"""


import os
import pandas as pd
from supabase import create_client
from tqdm import tqdm
from dotenv import load_dotenv
import time
import logging
import math

load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("VITE_SUPABASE_URL")
VITE_SUPABASE_ANON_KEY = os.getenv("VITE_SUPABASE_ANON_KEY")
TABLE = os.getenv("SUPABASE_TABLE", "recipes")
CSV_PATH = os.getenv("CSV_PATH")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))

logger.info(f"Configuration loaded:")
logger.info(f"  - Table: {TABLE}")
logger.info(f"  - CSV Path: {CSV_PATH}")
logger.info(f"  - Batch Size: {BATCH_SIZE}")

supabase = create_client(SUPABASE_URL, VITE_SUPABASE_ANON_KEY)


def parse_pg_array(x):
    """Accepte déjà le format postgres {a,b,c} ou une liste Python, renvoie une list Python."""
    if isinstance(x, list):
        return x
    if isinstance(x, float):
        return []
    x = str(x).strip()
    if x.startswith("{") and x.endswith("}"):
        inner = x[1:-1]
        if not inner:
            return []
        return [v.strip().strip('"') for v in inner.split(",")]
    return []


def main():
    logger.info("📥 Loading CSV...")
    df = pd.read_csv(filepath_or_buffer=CSV_PATH, index_col=0, sep=",")
    logger.info(f"✓ CSV loaded: {len(df)} rows, {len(df.columns)} columns")

    # Reset index pour convertir RecipeId en colonne
    df = df.reset_index()
    logger.info("✓ RecipeId index converted to column")

    # Convertir tous les noms de colonnes en minuscules
    df.columns = df.columns.str.lower()
    logger.info("✓ Column names converted to lowercase")

    array_cols = [
        "images",
        "keywords",
        "recipeingredientquantities",
        "recipeingredientparts",
    ]

    logger.info("🔧 Parsing array columns...")
    for col in array_cols:
        logger.debug(f"  Parsing column: {col}")
        df[col] = df[col].apply(parse_pg_array)
    logger.info(f"✓ Array columns parsed: {', '.join(array_cols)}")

    # Convertir les colonnes booléennes (int 0/1 -> bool)
    logger.info("🔄 Converting boolean columns...")
    bool_cols = [
        'is_vegan', 'is_vegetarian', 'contains_pork', 'contains_alcohol',
        'contains_gluten', 'contains_nuts', 'contains_dairy', 'contains_egg',
        'contains_fish', 'contains_soy', 'is_breakfast_brunch', 'is_dessert'
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    logger.info(f"✓ Boolean columns converted: {len([c for c in bool_cols if c in df.columns])} columns")

    # Convertir datepublished en timestamp proper
    logger.info("📅 Converting datepublished to timestamp...")
    if 'datepublished' in df.columns:
        df['datepublished'] = pd.to_datetime(df['datepublished']).astype(str)
        logger.info("✓ datepublished converted to datetime string")

    logger.info("🧹 Cleaning NaN values for JSON compatibility...")
    # Remplace NaN/inf par None pour TOUTES les colonnes numériques (exclut bool)
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns

    # Diagnostic: compte les NaN avant nettoyage
    nan_counts = df[numeric_cols].isna().sum()
    logger.info(f"  NaN values found in: {dict(nan_counts[nan_counts > 0])}")

    # Remplace NaN et inf par None
    for col in numeric_cols:
        df[col] = df[col].replace([float('inf'), float('-inf')], None)
        df[col] = df[col].where(pd.notna(df[col]), None)

    logger.info(f"✓ NaN/inf values cleaned for {len(numeric_cols)} numeric columns")

    logger.info("🚀 Starting upload to Supabase...")
    total_rows = len(df)
    total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"  Total rows: {total_rows}")
    logger.info(f"  Total batches: {total_batches}")

    successful_batches = 0
    failed_batches = 0

    for start in tqdm(range(0, total_rows, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, total_rows)
        batch_df = df.iloc[start:end]

        # Conversion en dict avec remplacement des NaN
        batch = batch_df.to_dict(orient="records")

        # Nettoie les NaN restants dans chaque record
        for record in batch:
            for key, value in list(record.items()):
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    record[key] = None

        batch_num = (start // BATCH_SIZE) + 1

        logger.debug(f"Processing batch {batch_num}/{total_batches} (rows {start}-{end})")

        # Retry logic
        for attempt in range(5):
            try:
                supabase.table(TABLE).insert(batch).execute()
                successful_batches += 1
                logger.debug(f"✓ Batch {batch_num} uploaded successfully")
                break
            except Exception as e:
                logger.warning(
                    f"⚠️ Error inserting batch {start}-{end} (attempt {attempt+1}/5): {str(e)}"
                )
                time.sleep(2**attempt)

                if attempt == 4:
                    failed_batches += 1
                    logger.error(f"❌ Fatal error on batch {batch_num} – stopping upload.")
                    logger.error(f"Upload summary: {successful_batches} successful, {failed_batches} failed")
                    return

    logger.info("\n✅ Upload finished successfully!")
    logger.info(f"Summary: {successful_batches}/{total_batches} batches uploaded successfully")


if __name__ == "__main__":
    main()
