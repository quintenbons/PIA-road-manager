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
        date="22 Janvier, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
      />
      <Title title="Génération de la carte" size="md" />
      <Paragraph text="Nous disposons d'outils pour obtenir des cartes sur lesquelles tester et entrainer notre IA." />
      <Title title="Génération par API" size="sm" />
      <Paragraph text="Nous disposons d'un outil permettant de récupérer les routes et intersections qui forment une ville. Nous décrivons ci-dessous le processus qui permet d'obtenir ce résultat:" />

      <ParagraphList
        paragraphs={[
          "L'API Overpass permet d'obtenir un ensemble de points représentant le réseau routier de la ville.",
          "À partir de ces points, nous reconstruisons les routes et identifions les intersections.",
          "Chaque route, définie par l'ensemble de points entre deux intersections, est simplifiée en un segment droit en conservant seulement les points extrêmes.",
          "Nous conservons uniquement le plus grand graphe connexe du réseau routier. Ceci élimine les petits graphes isolés qui résultent de l'exclusion des routes mineures (comme les parkings et les sorties résidentielles) lors de la récupération des données.",
        ]}
      />
      <Image src={GRENOBLE} width={"60%"} alignSelf={"center"} />
      <Paragraph text="Nous avons utilisé cette méthode pour récupérer la carte de Grenoble, puis, en utilisant Google Maps, nous avons supperposé notre résultat (en bleu et rouge) aux textures Google." />
      <Paragraph text="On constate quelques approximations, mais cette carte reste cohérente avec la réalité géographique." />
      <Paragraph text="Les cartes de villes et de villages sont malheureusement trop complexes et ne permettent pas d'isoler certains comportements. Voila pourquoi nous avons implémenté un second outil de génération." />

      <Title title="Génération avec le GUI" size="sm" />

      <Paragraph text="Il s'agit d'une interface graphique permettant de modéliser des cartes manuellement. Cela nous permet de tester notre moteur de simulation et de définir nos scénarios d'entrainement." />
      
      <Paragraph text="Les ressources suivantes présentent à gauche le GUI et à droite une simulation se basant sur la carte fraichement générée." />

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

      <Title title="Moteur de simulation" size="md" />
      <Paragraph text="Le but du moteur de pouvoir fournir un environnement physique virtuel à l'IA. Ce moteur physique doit pouvoir être assez performant pour entraîner l'IA dans un temps raisonnable, ce modèle devrait en théorie le plus se rapprocher de la réalité. Dans cette version, notre modèle est assez simpliste par rapport à la réalité. En revanche, il a été conçu pour pouvoir être amélioré sans grandes modifications de l'architecture du code. " />
      <Paragraph text="Le modèle physique implémente les notions suivantes:" />
      <ParagraphList
        paragraphs={[
          "Routes",
          "Intersections",
          "Objets qui se déplacent, ce sont les voitures",
          "Générateurs de trafic routier",
        ]}
      />
      
      <Paragraph text="Les générateurs de trafic produisent des voitures avec des itinéraires précalculés. Cet itinéraire est choisi pour être le plus court grâce à l'algorithme de Dijkstra. Les chemins possibles sont précalculés en utilisant une technologie plus rapide que Python (le C++)." />
      <Paragraph text="Par soucis de performance, la majeure partie du moteur a été réécrite en C++. Ainsi, le moteur est un mélange de C++ et de Python. Le C++ est utilisé pour les parties du code les plus exécutées. Ces parties correspondaient à plus de 99.5 % du temps passé pour la mise à jour et les calculs. C'est en partie ce qui nous à permis de générer suffisamment de dataset en un temps raisonnable." />
      <Paragraph text="Les voitures apparaissent et disparaissent sur des routes, un générateur de trafic se définit par une liste de routes d'apparition et une liste de routes de disparition. Ainsi que d'une fonction de génération qui prend en entrée un temps T et qui fait apparaître une voiture sur les routes d'apparition si la valeur de retour de cette fonction est supérieure à un." />
      <Paragraph text="Les routes font circuler les voitures. Les intersections contiennent la logique de signalisation et contrôlent le passage des voitures." />
      <Paragraph text="Les voitures adoptent un comportement dynamique inspiré de la conduite réelle, elles ralentissent et accélèrent pour respecter les limitations de vitesse. Dans les intersections, elles ralentissent et évitent à tout prix les collisions avec d'autres véhicules." />

      
      <Title title="Simulation graphique" size="md" />
      <Paragraph text="Au delas du moteur, nous disposons d'un outil de simulation graphique. Cet outil permet de visualiser la simulation en temps réel et de débogger plus simplement. Nous utilisons Pygame pour obtenir ces résulats." />

      <Paragraph text="La vidéo de gauche présente notre simulation graphique à la date du 15/12/2023, à droite, une version plus récente du 10/01/2024." />
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
          <iframe
          height={500}
          width={800}
          title="Graphical simulation"
          src="https://www.youtube.com/embed/RMmCcr4tUs4"
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
