import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.linspace(0, 10, 50)
y = 3 * x + 5 + np.random.randn(50) * 2

a = 0.0
b = 0.0
taux_apprentissage = 0.01
nb_iterations = 1000
n = len(x)

for i in range(nb_iterations):
    y_predit = a * x + b
    erreur = y_predit - y

    gradient_a = (2 / n) * np.sum(erreur * x)
    gradient_b = (2 / n) * np.sum(erreur)

    a = a - taux_apprentissage * gradient_a
    b = b - taux_apprentissage * gradient_b

print(f"Coefficients trouvés : a = {a:.2f}, b = {b:.2f}")

plt.scatter(x, y, label="Données")
plt.plot(x, a * x + b, color="red", label=f"Droite trouvée : y = {a:.2f}x + {b:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Régression linéaire par descente de gradient")
plt.legend()
plt.show()