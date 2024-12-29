import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.impute import SimpleImputer

#chargement de la base de donnée
df = pd.read_excel('base_de_donnees_nettoyee.xlsx')

#effectuer un lasso sur notre base de données

# Créer un objet imputer pour imputer les valeurs manquantes par la moyenne
imputer = SimpleImputer(strategy='mean')

# Appliquer l'imputation sur X
df_imputed =  pd.DataFrame(imputer.fit_transform(df.drop(["titre", 'Comédie', 'Comédie dramatique', 'Drame', 'Aventure', 'Animation', 'Famille', 'Thriller', 'Action', 'Péplum', 'Historique', 'Fantastique', 'Comédie musicale', 'Romance', 'Epouvante-horreur', 'Biopic', 'Musical', 'Science Fiction', 'Guerre', 'Policier', 'Espionnage', 'Western', 'Erotique', 'Arts Martiaux', 'Judiciaire', 'Expérimental', 'Bollywood', 'Évènement Sportif', 'Drama', 'Divers', 'Concert', 'Spectacle', 'Opéra'], axis=1)),columns=[ 'Presse', 'Spectateurs', 'durée', 'Année de production', 'nombre nationalités', 'prix', 'nominations', 'commercialiser', 'nombre_actrice', 'pressence_realisatrice', 'log_Budget', 'log_Box Office France'])
#'Comédie', 'Comédie dramatique', 'Drame', 'Aventure', 'Animation', 'Famille', 'Thriller', 'Action', 'Péplum', 'Historique', 'Fantastique', 'Comédie musicale', 'Romance', 'Epouvante-horreur', 'Biopic', 'Musical', 'Science Fiction', 'Guerre', 'Policier', 'Espionnage', 'Western', 'Erotique', 'Arts Martiaux', 'Judiciaire', 'Expérimental', 'Bollywood', 'Évènement Sportif', 'Drama', 'Divers', 'Concert', 'Spectacle', 'Opéra'
# Séparer les variables_explicatives de la variable_cible
v_explicatives = df_imputed.drop(columns=["Spectateurs"])
v_cible = df_imputed["Spectateurs"]

# Séparer les données en un jeu d'entraînement et de test
v_explicatives_train, v_explicatives_test, v_cible_train, v_cible_test = train_test_split(v_explicatives, v_cible, test_size=0.2, random_state=42)
print (v_explicatives_train.info())
print (v_explicatives_test.info())
print (v_cible_train.info())
print (v_cible_test.info())
# Standardiser les données (important pour la régression Lasso)
scaler = StandardScaler()
v_explicatives_train_scaled = scaler.fit_transform(v_explicatives_train)
v_explicatives_test_scaled = scaler.transform(v_explicatives_test)

# Créer le modèle Lasso
lasso = Lasso(alpha=0.1)  # Le paramètre alpha contrôle la régularisation (plus grand alpha => plus de régularisation)

# Entraîner le modèle Lasso
lasso.fit(v_explicatives_train_scaled, v_cible_train)
print(1)
# Prédictions sur les données de test
v_cible_pred = lasso.predict(v_explicatives_test)

# Évaluation du modèle
mse = mean_squared_error(v_cible_test, v_cible_pred)
r2 = r2_score(v_cible_test, v_cible_pred)

# Afficher les résultats
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Visualiser les coefficients
plt.figure(figsize=(10, 6))
plt.bar(v_explicatives.columns, lasso.coef_)
plt.title("Coefficients du modèle Lasso")
plt.show()

# Visualiser les prédictions par rapport aux vraies valeurs
plt.figure(figsize=(10, 6))
plt.scatter(v_cible_test, v_cible_pred)
plt.xlabel("Vraies valeurs")
plt.ylabel("Prédictions")
plt.title("Prédictions vs Vraies valeurs")
plt.show()