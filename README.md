# AI project: road management

## Quick start

`./src/main.py`

## Structure

- [examples/](./examples/): Examples and tests of simple sample usecases
- [src/](./src/)
  - [strategy](./src/strategy.py): AI strategy (TO START)
  - [engine/](./src/engine/): Engine only, try to keep this standalone so we can simulate without graphics too
  - [graphics/](./src/graphics/): All pygame related stuff, meaning it will rely on the engine

## 26/09

### Noeud
> Un élément qui sépare deux routes. Ex: Un passage piéton, un feu, un stop. Certains noeuds peuvent être gérés par le système d'IA (feux) tandis que d'autres ne servent qu'à la prise de déscision (autres que feux).

### Input
> Prétraitement transaparent au réseau de neuronne. Adaptable de part ce pré-traitement.

> - `hikers` : Combien a t-on de personnes sont à proximité du noeud (vélo/piéton) 
> - `vehicles` : Combien de véhicules sont à proximité du noeud
> - `hikers_speed` : Vitesse moyenne des `hikers` à proximité du noeud
> - `vehicles_speed` : Vitesse moyenne des `vehicles` à proximité du noeud
> - `hiker_queue_length` : Nombre de `hikers` en attente et à proximité du noeud
> - `vehicle_queue_length` : Nombre de `vehicles` en attente et à proximité du noeud

### Output
> Un booléen sur chaque neuds gérable par l'IA (noeuds représentant des feux).
