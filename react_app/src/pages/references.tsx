import { Container, theme } from "@chakra-ui/react";
import { BreadcrumbLnk } from "../components/BreadcrumbLnk";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Title } from "../components/Title";
import { AllLinks } from "../components/AllLinks";
import { Glossary } from "../components/Glossary";
import { text } from "stream/consumers";

export const References = (props: { setPath: (path: string) => void }) => {
  return (
    <Container
      maxW="container.lg"
      sx={{
        padding: theme.space[4],
        gap: theme.space[4],
        display: "flex",
        flexDirection: "column",
      }}
    >
      <BreadcrumbLnk
        setPath={props.setPath}
        sections={[
          {
            name: "Home",
            href: "/",
          },
          {
            name: "Références",
            href: "/references",
          },
        ]}
      />
      <DocumentDescriptor
        title="Références"
        date="13 Décembre, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
      />
      <Title title="Références" size="md" />

      <AllLinks
        links={[
          {
            text: "Github Ens'IA",
            url: "https://github.com/YannSia",
          },
          {
            text: "Medium - Simple CNN with numpy",
            url: "https://medium.com/analytics-vidhya/simple-cnn-using-numpy-part-i-introduction-data-processing-b6652615604d",
          },
          {
            text: "TP ens'IA",
            url: "https://gitlab.ensimag.fr/francoj/vorf",
          },
          {
            text: "PyTorch introduction (3h)",
            url: "https://youtu.be/c36lUUr864M?si=RYYo9ozEjq-OqBtU",
          },
          {
            text: "Overpass API Turbo",
            url: "http://overpass-api.de/",
          },
          {
            text: "Pygame",
            url: "https://www.pygame.org/news",
          },
        ]}
      />

      <Title title="Glossaire" size="md" />

      <Glossary
        words={[
          {
            text: "Intelligence artificielle",
            definition:
              "Domaine de l'informatique visant à créer des systèmes capables d'effectuer des tâches qui nécessitent normalement l'intelligence humaine.",
          },
          {
            text: "Données d'entrée",
            definition:
              "Ensemble de données utilisé pour former un modèle d'intelligence artificielle, fournissant des exemples sur lesquels le modèle apprend à effectuer une tâche spécifique.",
          },
          {
            text: "Optimisations locales",
            definition:
              "Techniques visant à améliorer la performance ou l'efficacité d'un modèle d'intelligence artificielle dans des régions spécifiques, sans nécessairement améliorer l'ensemble du modèle.",
          },
          {
            text: "Réseau neuronal dense",
            definition:
              "Type de réseau neuronal où chaque neurone est connecté à tous les neurones de la couche suivante, favorisant la transmission d'informations globales.",
          },
          {
            text: "Hidden layers",
            definition:
              "Couches intermédiaires d'un réseau neuronal où les transformations non linéaires des données se produisent, permettant au modèle d'apprendre des représentations complexes.",
          },
          {
            text: "Heuristique",
            definition:
              "Méthode basée sur l'expérience ou des règles pratiques pour résoudre un problème, souvent utilisée dans la conception d'algorithmes pour trouver des solutions efficaces.",
          },
          {
            text: "API",
            definition:
              "Interface de programmation d'application, définissant les règles et les outils permettant à des logiciels distincts de communiquer entre eux.",
          },
          {
            text: "GUI",
            definition:
              "Interface graphique utilisateur, une méthode d'interaction visuelle avec un logiciel ou un système informatique.",
          },
          {
            text: "Datasets",
            definition:
              "Ensembles de données structurées utilisés pour entraîner, évaluer ou tester des modèles d'intelligence artificielle.",
          },
          {
            text: "Dataloaders",
            definition:
              "Modules ou fonctions qui facilitent le chargement et la préparation des données pour l'entraînement des modèles d'intelligence artificielle.",
          },
          {
            text: "Modèle de Pathfinding",
            definition:
              "Algorithme ou modèle utilisé pour trouver le chemin le plus efficace entre deux points dans un environnement, souvent utilisé dans la planification de trajectoires pour les robots ou les jeux vidéo.",
          },
          {
            text: "Benchmark",
            definition:
              "Mesure de performance comparative servant à évaluer les capacités d'un système, d'un logiciel ou d'un composant par rapport à d'autres références.",
          },
        ]}
      />
    </Container>
  );
};
