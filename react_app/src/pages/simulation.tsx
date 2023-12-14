import { Container, theme, Image, Box } from "@chakra-ui/react";
import { BreadcrumbLnk } from "../components/BreadcrumbLnk";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Title } from "../components/Title";
import { Paragraph } from "../components/Paragraph";
import { ParagraphList } from "../components/ParagraphList";
import { GRENOBLE, GUI_NETWORK, NETWORK } from "../assets";
import { ContinueLectureButton } from "../components/ContinueLectureButton";

export const Simulation = (props: { setPath: (path: string) => void }) => {
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
            name: "Simulation",
            href: "/simulation",
          },
        ]}
      />
      <DocumentDescriptor
        title="Simulation"
        date="13 Décembre, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
      />
      <Title title="Génération de la carte" size="md" />
      <Title title="API" size="sm" />
      <Paragraph text="Nous disposons d'un outil permettant de récupérer la carte d'une ville." />
      <ParagraphList
        paragraphs={[
          "Récupération des Données : Nous utilisons l'API de Overpass pour obtenir un ensemble de points représentant le réseau routier de la ville.",
          "Reconstruction des Routes : À partir de ces points, nous reconstruisons les routes et identifions les intersections.",
          "Simplification des Routes : Chaque route, définie par l'ensemble de points entre deux intersections, est simplifiée en un segment droit en conservant seulement les points extrêmes.",
          "Sélection du Plus Grand Graphe Connexe : Nous conservons uniquement le plus grand graphe connexe du réseau routier. Ceci élimine les petits graphes isolés qui résultent de l'exclusion des routes mineures (comme les parkings et les sorties résidentielles) lors de la récupération des données.",
        ]}
      />
      <Image src={GRENOBLE} width={"60%"} alignSelf={"center"} />
      <Paragraph text="La carte générée de Grenoble, bien que présentant quelques approximations, reste largement cohérente avec la carte réelle de la ville. Par exemple, bien que la route traversant le CEA au Nord-Ouest soit légèrement décalée, notre modèle reproduit avec une précision raisonnable l'architecture routière. Ces modifications d'échelle sont dues aux approximations inhérentes à notre méthode de simplification, mais elles n'affectent pas l'intégrité globale du complexe routier. Cette fidélité à la structure réelle des routes est cruciale pour notre objectif, qui est de simuler de manière réaliste la gestion du trafic urbain." />

      <Title title="GUI" size="sm" />

      <Paragraph text="Nous avons développé une interface graphique pour modéliser nous même des cartes. Cela nous permet de tester notre moteur de simulation et de définir nos scénarios d'entrainement." />

      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <Image
          src={GUI_NETWORK}
          flex={"1"}
          width={"50%"}
          alignSelf={"center"}
        />
        <Image src={NETWORK} flex={"1"} width={"50%"} alignSelf={"center"} />
      </Box>

      <Title title="Engine" size="md" />
      <Paragraph text="Le modèle physique se découpe en plusieurs parties:" />
      <ParagraphList
        paragraphs={[
          "Des routes",
          "Des intersections",
          "Des objets qui se déplacent, pour la suite, on désignera cela par les voitures",
          "Des générateurs de trafic",
        ]}
      />
      <Paragraph text="Le générateur de trafic génère une voiture, avec un itinéraire. L'itinéraire est choisi avec un calcul de Dijkstra. Pour accélérer la simulation, l'ensemble des calculs de chemin sont précalculés en utilisant une techno plus rapide que le Python ici le C++." />
      <Paragraph text="Les routes doivent permettre de faire circuler des voitures et les intersections permettent de bloquer ou de faire passer les voitures." />
      <Paragraph text="Les principales difficultés sont liées au comportement des voitures. Il faut pouvoir ralentir en fonction de la vitesse des autres voitures, il faut pouvoir dépasser une voiture trop lente." />
      <Paragraph text="Nous avons aussi imaginé un système de collision qui permet aux voitures de réguler leurs allures en fonction du trafic et de traverser les intersections de manière réaliste." />

      <Title title="Simulation graphique" size="md" />
      <Paragraph text="La simulation graphique se fait avec la bibliothèque Pygame. Elle permet de visualiser la simulation en temps réel." />

      <Paragraph text="Voici une vidéo explicative de la simulation graphique:" />

      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <iframe
          height={500}
          width={800}
          title="Graphical simulation"
          src="https://www.youtube.com/embed/Hei4Z9-AF-I"
          allowFullScreen
        />
      </Box>

      <ContinueLectureButton
        text="Continuer vers Modèle"
        href="/modele"
        setPath={props.setPath}
      />
    </Container>
  );
};
