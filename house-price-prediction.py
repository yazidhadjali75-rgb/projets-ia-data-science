import numpy as np
import matplotlib.pyplot as plt

# Données : surface (m²) et prix (€) de 10 maisons
surfaces = np.array([50, 60, 65, 70, 80, 85, 90, 100, 110, 120])
prix = np.array([150000, 175000, 180000, 200000, 220000, 230000, 250000, 270000, 290000, 310000])

# Entraînement du modèle par descente de gradient
a = 0.0
b = 0.0
taux_apprentissage = 0.0001
nb_iterations = 10000
n = len(surfaces)

for i in range(nb_iterations):
    prix_predit = a * surfaces + b
    erreur = prix_predit - prix

    gradient_a = (2/n) * np.sum(erreur * surfaces)
    gradient_b = (2/n) * np.sum(erreur)

    a = a - taux_apprentissage * gradient_a
    b = b - taux_apprentissage * gradient_b

print(f"Modèle trouvé : prix = {a:.2f} * surface + {b:.2f}")

# Prédiction pour une nouvelle maison
nouvelle_surface = 95
prix_estime = a * nouvelle_surface + b
print(f"Prix estimé pour une maison de {nouvelle_surface} m² : {prix_estime:.2f} €")

# Affichage graphique
plt.scatter(surfaces, prix, label="Maisons connues")
plt.plot(surfaces, a * surfaces + b, color="red", label="Modèle de prédiction")
plt.scatter(nouvelle_surface, prix_estime, color="green", s=100, label="Nouvelle prédiction")
plt.xlabel("Surface (m²)")
plt.ylabel("Prix (€)")
plt.title("Prédiction de prix immobilier par régression linéaire")
plt.legend()
plt.show()