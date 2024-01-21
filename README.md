# AI project: road management

## Quick start

Avant tout:
  ```bash
  cd src/maps/cpp && make all
  pip install numpy torch pygame tqdm
  ```

#### Recompiler le simulateur

Cette étape est necessaire seulement si on doit toucher au code C++, ou si le binaire existant n'est pas compatible avec votre architechture.
Installer pybind (seulement ce qui est necessaire):

  ```bash
  pip install pybind
  apt install pybind11-dev
  ```

Compiler (une version compilée est déjà présente dans le repo)
  ```bash
  cd src/cpp && ./compile_cpp.sh
  ```

#### Créer une map

On a besoin de deux fichiers, une map et la liste des chemins correspondants.
S'assurer d'être à la racine du repo.

- Via l'API: Conforme au réseau routier réel d'une ville de France
  ```bash
  python3 src/maps/create_map_API.py <Nom_de_la_ville>
  ```
  puis suivre les instructions. La map a utilisé sera stockée dans `src/maps/build/API/<Nom_de_la_ville>/map.csv` et les chemins dans `src/maps/build/API/<Nom_de_la_ville>/paths.csv`.

- Via l'interface graphique: Libre placement des routes et des noeuds
  ```bash
  python3 src/maps/create_map_GUI.py
  ```
  - Cliquez pour ajouter un node.
  - Enfoncez `Espace` tout en cliquant sur 2 nodes pour créer une route bidirectionnelle entre eux.
  - Enfoncez `d` tout en cliquant sur un node pour le supprimer ainsi que toutes les routes qui y sont liées.

  La map a utilisé sera stockée dans `src/maps/build/GUI/<nom_choisi>/map.csv` et les chemins dans `src/maps/build/GUI/<nom_choisi>/paths.csv`.
  Attention à bien faire un graph totalement connecté.

  Options:
  - `--png`: Chemin vers une image PNG à utiliser comme arrière-plan.
  - `--grid`: Afficher la grille sur l'arrière-plan.

#### Lancer une simulation

Des cartes sont déjà disponibles. Nous vous conseillons de commencer par `src/maps/build/GUI/Training-4/Uniform/map.csv`.

```bash
  python3 src/main.py --map_file src/maps/build/GUI/<nom>/map.csv 
```

- `--nb_movables`: Nombre de véhicules à placer
- `--map_file`: Chemin vers le fichier de la map
- `--paths_file`: Chemin vers le fichier des chemins, si non spécifié, on prendra le fichier `paths.csv` dans le même dossier que `map.csv`

#### Générer des datasets d'entraînement

Les scripts utilisent argparse, donc `--help` pour plus d'infos.

- Localement: `python3 src/gen_dataset.py --size 10 src/maps/build/GUI/Training-4/Uniform`
- En parallèle: `python3 src/gen_dataset_parallel.py`
- Avec les vmgpu de l'ENSIMAG (attention à ne pas gêner les autres): `./training/gen_datasets.sh`

## Structure

- [data](./data/): Datasets, modèles, etc.
- [docs](./docs/): Aide à la documentation
- [react_app](./react_app/): Rendu
- [training](./training/): Scripts de génération/entrainement
- [src/](./src/)
  - [cpp/](./src/cpp/): Code C++ pour la simulation
  - [engine/](./src/engine.py): Code python pour la simulation (binding C++)
  - [graphics/](./src/graphics.py): Code totalement isolé pour l'affichage (pygame)
  - [maps/](./src/maps/): Code pour la génération de maps
  - [ai/](./src/ai/): Code pour l'IA (pytorch)
  - *.py: multitudes de scripts pour l'entrainement, la génération de datasets, la simulation en direct...

## 26/09 (Idée initiale)

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

## Intelligence artificielle

- Générer un dataset

`src/gen_dataset.py -h`

créera un fichier `dataset.pt`

- Entraîner un modèle

`src/train_ai.py -h`

créera un fichier `model.pt` SI BESOIN (sinon ça le charge juste), et l'entraînera. Notez qu'il n'est pas save après l'entraînement pour l'instant
