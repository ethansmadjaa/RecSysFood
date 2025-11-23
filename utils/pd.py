"""
This module is used to process the data and convert it to a format that can be used by the Supabase database.
"""


import pandas as pd
import numpy as np


df = pd.read_csv("cleaned_and_labeled_recipes.csv", index_col=0, sep=",")

print(df.head())


def r_vector_to_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) or pd.isnull(x):
        return None

    x = str(x).strip()

    # Format invalide
    if not x.startswith("c(") or not x.endswith(")"):
        return None

    # Retirer c(  )
    inner = x[2:-1].strip()

    if inner == "" or not inner or pd.isna(inner) or pd.isnull(inner):
        return None

    # Parse en respectant les guillemets
    cleaned = []
    current = ""
    in_quotes = False
    quote_char = None

    for i, char in enumerate(inner):
        if char in ('"', "'") and (i == 0 or inner[i-1] != '\\'):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
            else:
                current += char
        elif char == ',' and not in_quotes:
            # On a trouvé une virgule hors des guillemets
            value = current.strip().strip('"').strip("'").strip()
            if value and value != "NA":
                cleaned.append(value)
            current = ""
        else:
            current += char

    # Ajouter le dernier élément
    value = current.strip().strip('"').strip("'").strip()
    if value and value != "NA":
        cleaned.append(value)

    if len(cleaned) == 0:
        return None

    return cleaned


array_cols = [
    "Images",
    "Keywords",
    "RecipeIngredientQuantities",
    "RecipeIngredientParts",
    "RecipeInstructions",
]

for col in array_cols:
    print(col + " : " + "is being converted")
    df[col] = df[col].apply(r_vector_to_list)


print(df.info())  # 1. Types de colonnes + non-null count
print(df.head(5))  # 2. Aperçu des 5 premières lignes
print(df.dtypes)  # 3. Types pandas bruts
print(df.apply(lambda x: type(x.iloc[0])))  # 4. Type Python de chaque cellule
print(df.sample(3))


df.reset_index(inplace=True)
df.rename(columns={"index": "RecipeId"}, inplace=True)


for col in array_cols:
    print(col + " : " + "is being sampled")
    for item in df[col].sample(5):
        print(item)


# Convert boolean columns (0/1 to False/True)
bool_cols = [
    "Is_Vegan",
    "Is_Vegetarian",
    "Contains_Pork",
    "Contains_Alcohol",
    "Contains_Gluten",
    "Contains_Nuts",
    "Contains_Dairy",
    "Contains_Egg",
    "Contains_Fish",
    "Contains_Soy",
    "Is_Breakfast_Brunch",
    "Is_Dessert",
]

for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(bool)



df.to_csv("recipes.csv", index=False)
