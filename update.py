import ast
import pandas as pd
from lib import supabase


def convert_python_list_to_postgres_array(value):
    """Convert Python list string representation to actual list for Supabase."""
    if pd.isna(value):
        return None

    # If it's already a list, return it
    if isinstance(value, list):
        return value

    # If it's a string representation of a list, parse it
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            # Use ast.literal_eval to safely parse the Python list string
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass

    return value


print("Loading recipes.csv...")
df = pd.read_csv("utils/recipes.csv", sep=",", encoding="utf-8")
df.columns = df.columns.str.lower()
print(f"Loaded {len(df)} recipes")
columns_to_update = [
    "images",
    "keywords",
    "recipeingredientquantities",
    "recipeingredientparts",
]

# Create a new dataframe with only the columns we need
new_df = df[["recipeid"] + columns_to_update].copy()

# Convert Python list strings to actual lists for all columns
for column in columns_to_update:
    df[column] = df[column].apply(convert_python_list_to_postgres_array)
    new_df[column] = new_df[column].apply(convert_python_list_to_postgres_array)

print("updated new_df")
df.to_csv("utils/recipes.csv", index=False)

# Process in batches for better performance
BATCH_SIZE = 1000
SKIP_FIRST = 52000  # Skip the first 31000 recipes
total_rows = len(new_df)

print(f"Skipping first {SKIP_FIRST} recipes, starting from recipe {SKIP_FIRST + 1}")

for i in range(SKIP_FIRST, total_rows, BATCH_SIZE):
    batch = new_df.iloc[i : i + BATCH_SIZE]
    print(
        f"Processing batch {(i - SKIP_FIRST) // BATCH_SIZE + 1}/{(total_rows - SKIP_FIRST) // BATCH_SIZE + 1}"
    )
    for _, row in batch.iterrows():
        # Build update dict with only non-null values
        update_data = {}
        for column in columns_to_update:
            value = row[column]
            if value is not None:
                update_data[column] = value

        if update_data:
            try:
                supabase.table("recipes").update(update_data).eq(
                    "recipeid", int(row["recipeid"])
                ).execute()
            except Exception as e:
                print(f"Error updating recipe {row['recipeid']}: {e}")

    print(f"Processed {min(i + BATCH_SIZE, total_rows)}/{total_rows} recipes")

print("All columns updated successfully")
