# Modèle d'Ising avec bruit

Projet de validation du cours "Simulation et Monte-Carlo" de l'ENSAE (2e année)

Le sujet est reproduit ci-dessous:

> Une méthode classique pour débruiter une image est de considérer un modèle de la forme: $y_{i} \mid x_{i}=k \sim N\left(\mu_{k}, \tau^{2}\right)$, où $y_{i}$ est le niveau de gris du pixel $i$, et les $x_{i}$ (à valeurs dans $\{0,1\}$ ) suivent une loi dite d'Ising:
> $$
> p(x)=\frac{1}{Z(\alpha, \beta)} \exp \left\{\alpha \sum_{i} x_{i}+\beta \sum_{i \sim j} 1\left\{x_{i}=x_{j}\right\}\right\}
> $$
> où la deuxième somme porte sur les paires de voisins (deux pixels sont voisins s'ils ont un côté adjacent), et $Z(\alpha, \beta)$ est une constante de normalisation (difficile à évaluer, expliquer pourquoi).
> 1. On suppose $\tau, \alpha$ et $\beta$ connus. Mettre en oeuvre un Gibbs sampler pour simuler selon loi des $x_{i}$ sachant les $y_{i}$. On pourra l'appliquer à une version bruitée d'une image de votre choix. Par exemple: https://images.fineartamerica.com/images/artworkimages/mediumlarge/ 3/op-art-black-and-white-infinity-whirl-tom-hill.jpg
> 2. On suppose $\tau$ inconnu. Adapter l'algorithme de la question précédente pour estimer $\tau$. On pourra prendre une loi a priori de type inverse-gamma (expliquer pourquoi). Quels problèmes se posent si on choisit mal les hyper-paramètres de cette loi a priori?
> 3. On suppose $\alpha$ et $\beta$ eux aussi inconnus. Est-il possible de construire de généraliser à nouveau l'algorithme des questions précédentes pour estimer $\alpha$ et $\beta$ (en plus de $\tau$ et des variables $x_{i}$ ). (Attention, la réponse dépend de la loi a priori choisie). L'implementer si vous avez répondu par l'affirmative.
> 4. Bonus: proposer un algorithme de type ABC pour simuler la loi jointe de tous les paramètres et des $x_{i}$, sans contrainte sur la loi a priori de $(\alpha, \beta)$. Discuter et illustrer la performance de l'algorithme. En fonction du temps, vous pouvez aussi généraliser à des images avec $K$ couleurs (loi a priori de type Potts).