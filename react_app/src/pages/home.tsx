import { Container, theme, Image, Box } from "@chakra-ui/react";
import { Title } from "../components/Title";
import { Paragraph } from "../components/Paragraph";
import { ParagraphList } from "../components/ParagraphList";
import { MultipleTabs } from "../components/MultipleTab";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Banner } from "../components/Banner";
import {
  CROSS_DUPLEX,
  OPEN,
  OPEN_CORRIDOR,
  PIECE_OF_CAKE,
  STRATEGY_GRAPH_EXAMPLE,
  SUMMARY_DRAWING,
} from "../assets";
import { ContinueLectureButton } from "../components/ContinueLectureButton";

export const Home = (props: { setPath: (path: string) => void }) => {
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
      <Banner />
      <DocumentDescriptor
        title="Gestion du trafic par IA"
        date="22 Janvier, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
      />
      <Title title="Architecture Préliminaire de l'IA" size="lg" />      
      <Title title="Introduction" size="md" />
      <Paragraph text="Ceci est le rapport du projet IA. Le github est accessible depuis la navbar." />
      <Paragraph text="La simulation du réseau routier est un projet complexe visant à simuler fidèlement les conditions de circulation urbaine. Initialement, notre projet envisageait de créer une intelligence artificielle unique ayant la maîtrise sur l'ensemble des feux de signalisation de la ville, décidant du comportement de chaque feu. Cependant, cette méthode a soulevé deux problématiques majeures : premièrement, la génération des données d'entrée était excessivement longue, rendant cette approche peu pratique compte tenu du volume des données requises. Deuxièmement, l'ampleur des données d'entrée soulevait des inquiétudes quant à la capacité du système à converger vers une solution efficace." />
      <Paragraph text="Pour pallier à ces défis, nous avons modifié notre stratégie. Plutôt que d'avoir une seule IA pour tout le réseau, notre système se compose désormais d'une multitude d'IA, chacune gérant un noeud spécifique du réseau, c'est-à-dire une intersection avec ses feux de signalisation. Ainsi, chaque IA est responsable de tous les feux d'une intersection donnée et doit optimiser la circulation à ce point précis. Les noeuds reçoivent des flux de véhicules sur les routes entrantes et chaque IA s'efforce de minimiser la congestion en contrôlant efficacement les feux de son intersection." />
      <Paragraph text="Cette nouvelle méthode a pour objectif de trouver une optimisation locale pour chaque scénario, qui dépend de divers facteurs comme le flux de voitures, le nombre de routes connectées à l'intersection, et les caractéristiques propres à chaque route. Nous espérons ainsi que les IA trouvent les optimisations locales, ce qui (nous l'espérons) contribuera à une amélioration globale et coordonnée de la circulation sur l'ensemble du réseau routier." />
      <Title title="Flux d'utilisation" size="md" />
      <Image src={SUMMARY_DRAWING} />
      <Paragraph text="L'intelligence artificielle utilisée est un réseau neuronal dense à 4 étages. Il utilisera donc 2 'hidden layers', chacun constitué de 64 neuronnes." />
      <ParagraphList
        paragraphs={[
          "L'entrée correspond à l'état d'une intersection, ainsi que des évènements issus de la simulation sur les 15 minutes précédentes. Il est important d'inclure dans cette entrée la stratégie précédemment choisie, ainsi que la congestion (terme qui n'est pas encore défini mathématiquement dans notre modèle).",
          "La sortie correspond à un softmax des scores prévus pour les stratégies disponibles. Les stratégie sont des algorithmes de gestion de la circulation conçus par nos soins. Elles sont décrite dans la suite du rapport. Il est important ici d'avoir des stratégies cohérentes avec le monde réel, mais aussi adaptables au nombre d'entrées/sorties de l'intersection sur laquelle on l'utilise.",
        ]}
      />
      <Paragraph text="Cela signifie que l'IA aura un impact uniquement toutes les 15 minutes, et qu'elle n'a d'informations que sur une seule intersection. Cela nous semblait raisonable sous quelques hypothèses:" />
      <ParagraphList
        paragraphs={[
          "Le modèle du trafic ne varie pas beaucoup en 15 minutes.",
          "Une intersection est affectée par les intersections environnentes, mais il est possible de très bien gérer une intersection sans avoir à communiquer avec les intersections voisines.",
        ]}
      />
      <Title title="Les stratégies" size="md" />
      <Paragraph text="Les stratégies sont des algorithmes qui permettent de gérer les feux de signalisation d'une intersection. Nous utilisons le paradigme de stratégie afin de restreindre les choix de l'IA sur un sous-ensemble de solutions possibles dont on sait qu'une au moins sera proche de la meilleure solution. Cette approche heuristique permet des résultats cohérents avec un coût en temps et en énergie raisonable."/>
      <Paragraph text="Jusqu'à présent nous avons dégagé un total de quatre stratégies primaires. Les règles de circulation classiques telles que les priorités à droite ou les distances de sécurité s'appliquent en priorité sur les stratégies." />
      <Paragraph text="Chaque stratégie peut être représentée par un graphe d'état cyclique. Les arrêtes reliants chaque état correspondent au temps nécessaires pour passer à l'état suivant." />
      <Image src={STRATEGY_GRAPH_EXAMPLE} width={"60%"} alignSelf={"center"} />
      <Paragraph text="Ce schéma présente une intersection avec deux feux de signalisation. Cette intersection se traduit en un graphe à deux états, l'état n°1 représente la route avec la signalisation bleue ouverte et avec la signalisation orange bloquée. L'état n°2 inverse les routes bloquées et ouvertes." />
      <Paragraph text="Parmi les stratégies, nous générons aussi des mutations de celles-ci qui rendent les transitions plus ou moins longues. Cela a pour but de donner à l'IA de la flexibilité et de s'adapter lorsque les routes d'entrée des carrefours ont des débits de véhicules différents." />
      
      <Title title="Présentation des stratégies primaires" size="sm" />
      <Paragraph text="Ce composant présente les différentes stratégies implémentées. Les voitures sont représentées par les points de couleur, les intersections par les larges points verts. Une route bloquée est une route dont la couleur est noire." />
      
      <MultipleTabs
        childrens={{
          Open: (
            <Box
              sx={{
                display: "flex",
                flexDirection: "row",
                gap: theme.space[4],
              }}
            >
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignSelf: "center",
                  gap: theme.space[4],
                }}
              >
                <Paragraph text="La stratégie 'Open' consiste à laisser tous les feux au vert. De cette façon le trafic se régule de manière autonome." />
                <Paragraph text="Dans l'exemple suivant, les flux possibles sont représentés en orange." />
                <Paragraph text="Les stratégies peuvent changer au cours du temps, en pratique un feu de signalisation reste en place, choisir cette stratégie revient à faire en sorte que tous les feux de signalisation de l'intersection soient oranges et clignotent." />
              </Box>
              <Image src={OPEN} />
            </Box>
          ),
          "Open Corridor": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "row",
                gap: theme.space[4],
              }}
            >
              <Box
                sx={{
                  alignSelf: "center",
                  display: "flex",
                  flexDirection: "column",
                  gap: theme.space[4],
                }}
              >
                <Paragraph text="La stratégie 'Open corridor' correspond à un feu qui restera toujours vert tandis que les autres feux de signalisation alterneront chacun leur tour à temps égal." />
                <Paragraph text="Dans l'exemple suivant, le feu de signalisation qui restera en vert est aussi représenté par cette couleur tandis que les feux rouges et bleus alternent." />
              </Box>
              <Image src={OPEN_CORRIDOR} />
            </Box>
          ),
          "Piece of Cake": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "row",
                gap: theme.space[4],
              }}
            >
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignSelf: "center",
                  gap: theme.space[4],
                }}
              >
                <Paragraph text="La stratégie 'Piece of cake': si on représente le temps total disponible avec un gâteau, cette stratégie partage équitablement le gâteau entre tous les feux de signalisation. Ainsi, lorsqu'un feu est au vert, les autres sont au rouge." />
                <Paragraph text="Dans l'exemple suivant les feux de signalisations sont représentés par les traits de couleur rouge, vert et bleu." />
              </Box>
              <Image src={PIECE_OF_CAKE} />
            </Box>
          ),
          "Cross Duplex": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "row",
                gap: theme.space[4],
              }}
            >
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignSelf: "center",
                  gap: theme.space[4],
                }}
              >
                <Paragraph text="La stratégie 'Cross duplex' est la plus intuitive. Elle correspond au carrefour typique et permet de faire alterner les feux de signalisation des routes parallèles avec ceux des routes perpendiculaires." />
                <Paragraph text="Dans l'exemple suivant sont représentés en bleu et rouge les groupes de feux de signalisation. Le groupe bleu et le groupe rouge s'allument chacun leur tour. " />
              </Box>
              <Image src={CROSS_DUPLEX} />
            </Box>
          ),
        }}
      />

      <ContinueLectureButton
        text="Continuer vers Simulation"
        href="/simulation"
        setPath={props.setPath}
      />
    </Container>
  );
};
