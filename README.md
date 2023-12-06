# AI project: road management

## Quick start

Avant tout:
  ```bash
  cd src/maps/cpp && make all
  ```

#### Créer une map
On a besoin de deux fichiers, une map et la liste des chemins correspondants.
S'assurer d'être au root du repo.

- Via l'API:
  ```bash
  python3 src/maps/create_map_API.py <Nom_de_la_ville>
  ```
  puis suivre les instructions. La map a utilisé sera stockée dans `src/maps/build/API/<Nom_de_la_ville>/map.csv` et les chemins dans `src/maps/build/API/<Nom_de_la_ville>/paths.csv`.

- Via l'interface graphique:
  ```bash
  python3 src/maps/create_map_GUI.py 
  ```
  - On clic pour ajouter un node.
  - On reste appuyer sur `Espace` tout en cliquant sur 2 nodes pour créer une route entre eux.
  - On reste appuyer sur `d` tout en cliquant sur un node pour le supprimer ainsi que toutes les routes qui y sont liées.

  La map a utilisé sera stockée dans `src/maps/build/GUI/<nom_choisi>/map.csv` et les chemins dans `src/maps/build/GUI/<nom_choisi>/paths.csv`.
  Attention à bien faire un graph totalement connecté.

#### Lancer une simulation

```bash
  python3 src/main.py --nb_movables <int> --map_file src/maps/build/GUI/<nom>/map.csv 
```

- `--nb_movables`: Nombre de véhicules à placer
- `--map_file`: Chemin vers le fichier de la map
- `--paths_file`: Chemin vers le fichier des chemins, si non spécifié, on prendra le fichier `paths.csv` dans le même dossier que `map.csv`

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

## Modèle

`road`: Représente les routes, elles peuvent être bidirectionnelles. Les véhicules sont sur des routes.

`vehicle`: Représente un véhicule, si la position est à 1 ou plus, cela signifie que le véhicule doit passer sur la prochaine route sur son chemin, à condition qu'il ne soit pas bloqué.

`trafficFlow`: 

Une `simulation` contient des `node` représentant des intersections ainsi que des `car` représentant des véhicules.
La `simulation` a une méthode `run` qui permet de jouer la `simulation`. Lorsqu'une `simulation` est jouée, elle met à jour les états des `node` et des `car` de manière itérative et jusqu'à ce que toutes les `car` aient atteinte leur objectif.

Lorsqu'une `car` se met à jour, elle commence par calculer sa vitesse puis sa nouvelle position.
La position `car` évolue se situe entre 0 et 1, 1 signifie qu'elle a atteint une bordure de la `road` sur laquelle elle se trouve.
- Si elle atteint une bordure de `road`, on va chercher dans le `node` comment atteindre la `next_road`. Pour atteindre la `next_road`, la `car` devra traverser un `FlowController` qui fait la jointure entre les deux. Ce `FlowController` peut être ouvert ou fermé. Il ne sera traversé que si il est ouvert. Dans le cas ou celui-ci est ouvert, la `car` pourra réinitialiser sa position et se placer sur la prochaine `road`.

Lorsqu'un `node` se met à jour, il met à jour tout ses `FlowController`. Les `FlowController` ont chacun leur propre logique pour se mettre à jour.
- `TrafficLight`: Les états sont gérés par l'IA donc rien n'est mis à jour, cependant si le `TrafficLight` dispose d'un `Crosswalk`, celui ci sera mit à jour en fonction de l'état du `TrafficLight`.
- `Crosswalk`: Dons le cas d'un `Crosswalk` classique, si des piétons attendent ils seront autorisés à traverser. Si ils traversent, le `Crosswalk` sera bloqué pendant `CROSSWALK_SPEED` unités de temps. Sinon le `Crosswalk` est ouvert.
