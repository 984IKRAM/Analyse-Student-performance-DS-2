import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Configuration de Matplotlib pour des graphiques clairs
plt.style.use('ggplot')
sns.set_style('whitegrid')

# --- 1. Chargement des données ---
FILE_PATH = 'StudentsPerformance.csv'
TARGET_COLUMN = 'math score'

try:
    df = pd.read_csv(FILE_PATH)
    print(f"Dataset '{FILE_PATH}' chargé avec succès. {len(df)} lignes trouvées.")
except FileNotFoundError:
    print(f"Erreur: Le fichier '{FILE_PATH}' n'a pas été trouvé. Veuillez vérifier le chemin.")
    exit()

# --- 2. Nettoyage et Préparation (EDA et Encodage) ---
print("\n--- 2. Préparation des données ---")

# Renommage des colonnes pour la simplicité
df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_').str.lower()
TARGET_COLUMN = 'math_score'

# Variables Catégorielles à Encoder (pour les modèles)
categorical_cols = [
    'gender', 
    'race_ethnicity', 
    'parental_level_of_education', 
    'lunch', 
    'test_preparation_course'
]

le = LabelEncoder()
df_encoded = df.copy()

for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    print(f"  - Colonne '{col}' encodée par LabelEncoder.")

# --- 3. Analyse Exploratoire (EDA) et Visualisation ---

print("\n--- 3. Analyse Exploratoire (EDA) ---")

# 3.1 Matrice de Corrélation
# On utilise les scores et les variables encodées pour la corrélation
corr_matrix = df_encoded[['math_score', 'reading_score', 'writing_score'] + categorical_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Coefficient de Corrélation'})
plt.title("Matrice de Corrélation des Scores et Facteurs", fontsize=16, weight='bold')
plt.show()
print("Graphique 1: Matrice de Corrélation affichée (Corrélation Score lecture/écriture > 0.90).")

# 3.2 Impact du Déjeuner
plt.figure(figsize=(8, 6))
sns.boxplot(x='lunch', y=TARGET_COLUMN, data=df)
plt.title("Impact du Déjeuner sur le Score en Mathématiques", fontsize=14)
plt.xlabel("Type de Déjeuner", fontsize=12)
plt.ylabel("Score en Mathématiques", fontsize=12)
plt.show()
print("Graphique 2: Boxplot Déjeuner vs Score Mathématiques affiché.")

# 3.3 Impact du Cours de Préparation
plt.figure(figsize=(8, 6))
sns.boxplot(x='test_preparation_course', y=TARGET_COLUMN, data=df)
plt.title("Impact du Cours de Préparation sur le Score en Mathématiques", fontsize=14)
plt.xlabel("Cours de Préparation", fontsize=12)
plt.ylabel("Score en Mathématiques", fontsize=12)
plt.show()
print("Graphique 3: Boxplot Cours de Préparation vs Score Mathématiques affiché.")


# --- 4. Modélisation de Régression ---

# Définition des Caractéristiques (X) et de la Cible (y) pour les modèles
X = df_encoded.drop(TARGET_COLUMN, axis=1) 
y = df_encoded[TARGET_COLUMN]

# Split des Données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nDonnées divisées : Train {X_train.shape[0]} samples, Test {X_test.shape[0]} samples.")

results = {}

# =========================================================================
# --- MODÈLE 1: RANDOM FOREST REGRESSOR ---
# =========================================================================

print("\n--- 4.1 Modélisation : Random Forest Regressor ---")

model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
results['RandomForest'] = {'R2': r2_rf, 'RMSE': rmse_rf}

print(f"Random Forest Regressor - R-squared (R²): {r2_rf:.4f}")
print(f"Random Forest Regressor - RMSE: {rmse_rf:.2f}")

# Visualisation Réel vs. Prédit (Random Forest)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.6, color='skyblue', edgecolor='black')
min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Prédiction Parfaite')
plt.title("Régression par Forêt Aléatoire : Scores Réels vs. Prédits", fontsize=14)
plt.xlabel("Score Réel en Mathématiques", fontsize=12)
plt.ylabel("Score Prédit en Mathématiques", fontsize=12)
plt.legend()
plt.show()
print("Graphique 4: Scatter Plot Random Forest affiché.")

# Importance des Variables (Random Forest)
feature_importance = model_rf.feature_importances_
feature_names = X.columns
sorted_idx = feature_importance.argsort()[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx], y=feature_names[sorted_idx], palette="viridis")
plt.title("Importance des Variables - Random Forest Regressor", fontsize=14)
plt.xlabel("Importance de la Caractéristique", fontsize=12)
plt.ylabel("Caractéristique", fontsize=12)
plt.tight_layout()
plt.show()
print("Graphique 5: Importance des Variables Random Forest affiché.")

# =========================================================================
# --- MODÈLE 2: SUPPORT VECTOR REGRESSOR (SVR) ---
# =========================================================================

print("\n--- 4.2 Modélisation : SVR (Support Vector Regressor) ---")

# Mise à l'échelle des données (nécessaire pour SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Caractéristiques mises à l'échelle (StandardScaler).")

# Initialisation et Entraînement du modèle SVR
model_svr = SVR(kernel='rbf')
model_svr.fit(X_train_scaled, y_train)
y_pred_svr = model_svr.predict(X_test_scaled)

r2_svr = r2_score(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
results['SVR'] = {'R2': r2_svr, 'RMSE': rmse_svr}

print(f"SVR Regressor - R-squared (R²): {r2_svr:.4f}")
print(f"SVR Regressor - RMSE: {rmse_svr:.2f}")

# Visualisation Réel vs. Prédit (SVR)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_svr, alpha=0.6, color='salmon', edgecolor='black')
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Prédiction Parfaite')
plt.title("Régression SVR (rbf) : Scores Réels vs. Prédits", fontsize=14)
plt.xlabel("Score Réel en Mathématiques", fontsize=12)
plt.ylabel("Score Prédit en Mathématiques", fontsize=12)
plt.legend()
plt.show()
print("Graphique 6: Scatter Plot SVR affiché.")

# --- 5. Synthèse des Résultats ---
print("\n--- 5. Synthèse des Résultats de Modélisation ---")
results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='R2', ascending=False).to_markdown(floatfmt=".4f"))
print("\nAnalyse et Modélisation Terminées.")
