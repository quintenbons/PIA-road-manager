import { Box, Container, theme, Image } from "@chakra-ui/react";
import { Title } from "../components/Title";
import { Paragraph } from "../components/Paragraph";
import { AllLinks } from "../components/AllLinks";
import { AccordionParagraph } from "../components/AccordionParagraph";
import { ParagraphList } from "../components/ParagraphList";
import { MultipleTabs } from "../components/MultipleTab";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Banner } from "../components/Banner";
import { BreadcrumbLnk } from "../components/BreadcrumbLnk";
import {
  CROSS_DUPLEX,
  DATASET_GENERATION,
  DATASET_HESITATION,
  OPEN,
  OPEN_CORRIDOR,
  PIECE_OF_CAKE,
  SUMMARY_DRAWING,
} from "../assets";
import { ContinueLectureButton } from "../components/ContinueLectureButton";

export const Home = () => {
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
        date="13 Décembre, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
      />
      <Title title="Architecture Préliminaire de l'IA" size="lg" />
      <Title title="Introduction" size="md" />
      <Paragraph text="La simulation du réseau routier de Grenoble est un projet complexe visant à simuler fidèlement les conditions de circulation urbaine. Initialement, notre projet envisageait de créer une intelligence artificielle unique ayant la maîtrise sur l'ensemble des feux de signalisation de la ville, décidant du comportement de chaque feu. Cependant, cette méthode a soulevé deux problématiques majeures : premièrement, la génération des données d'entrée était excessivement longue, rendant cette approche peu pratique compte tenu du volume des données requises. Deuxièmement, l'ampleur des données d'entrée soulevait des inquiétudes quant à la capacité du système à converger vers une solution efficace." />
      <Paragraph text="Pour pallier à ces défis, nous avons modifié notre stratégie. Plutôt que d'avoir une seule IA pour tout le réseau, notre système se compose désormais d'une multitude d'IA, chacune gérant un noeud spécifique du réseau, c'est-à-dire une intersection avec ses feux de signalisation. Ainsi, chaque IA est responsable de tous les feux d'une intersection donnée et doit optimiser la circulation à ce point précis. Les noeuds reçoivent des flux de véhicules sur les routes entrantes et chaque IA s'efforce de minimiser la congestion en contrôlant efficacement les feux de son intersection." />
      <Paragraph text="Cette nouvelle méthode a pour objectif de trouver une optimisation locale pour chaque scénario, qui dépend de divers facteurs comme le flux de voitures, le nombre de routes connectées à l'intersection, et les caractéristiques propres à chaque route. Nous espérons ainsi que les IA trouvent les optimisations locales, ce qui (nous l'espérons) contribuera à une amélioration globale et coordonnée de la circulation sur l'ensemble du réseau routier." />
      <Title title="Flux d'utilisation" size="md" />
      <Image src={SUMMARY_DRAWING} />
      <Paragraph text="L'intelligence artificielle utilisée est un réseau neuronal dense à 4 étages. Il utilisera donc 2 'hidden layers', chacun constitué de 64 neuronnes." />
      <ParagraphList
        paragraphs={[
          "L'entrée correspond à l'état d'une intersection, ainsi que des évènements issus de la simulation sur les 15 minutes précédentes. Il est important d'inclure dans cette entrée la stratégie précédemment choisie, ainsi que la congestion (terme qui n'est pas encore défini mathématiquement dans notre modèle).",
          "La sortie correspond à un choix de stratégie (one hot, choisi avec un softmax) à prendre pour les 15 prochaines minutes pour une intersection donnée. Cette stratégie est un algorithme conçu par nos soins. Il est important ici d'avoir des stratégies cohérentes avec le monde réel, mais aussi adaptables au nombre d'entrées/sorties de l'intersection sur laquelle on l'utilise.",
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
      <Paragraph text="Jusqu'à présent nous avons dégagé un total de quatre stratégies primaires. Ces stratégies représentent les logiques s'appliquant aux feux. Les règles de circulation classiques telles que les priorités à droite ou les distances de sécurité s'appliquent en priorité sur les stratégies." />
      <Paragraph text="Chaque stratégie peut être représentée par un graphe d'état cyclique. Nous pensons faire des variantes de ces graphes dans lesquelles l'IA pourra choisir le temps nécessaire pour passer à l'état suivant. Cela a pour but de donner à l'IA de la flexibilité et de s'adapter lorsque les routes d'entrée des carrefours ont des débits de véhicules différents." />
      <MultipleTabs
        childrens={{
          Open: (
            <Container
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="La stratégie 'Open' consiste à laisser tous les feux au vert. De cette façon le trafic se régule de manière autonome." />
              <Paragraph text="Dans l'exemple suivant, les flux possibles sont représentés en orange." />
              <Image src={OPEN} />
              <Paragraph text="Les stratégies peuvent changer au cours du temps, en pratique un feu de signalisation reste en place, choisir cette stratégie revient à faire en sorte que tous les feux de signalisation de l'intersection soient oranges et clignotent." />
            </Container>
          ),
          "Open Corridor": (
            <Container
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="La stratégie 'Open corridor' correspond à un feu qui restera toujours vert tandis que les autres feux de signalisation alterneront chacun leur tour à temps égal." />
              <Paragraph text="Dans l'exemple suivant, le feu de signalisation qui restera en vert est aussi représenté par cette couleur tandis que les feux rouges et bleus alternent." />
              <Image src={OPEN_CORRIDOR} />
            </Container>
          ),
          "Cross Duplex": (
            <Container
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="La stratégie 'Piece of cake': si on représente le temps total disponible avec un gâteau, cette stratégie partage équitablement le gâteau entre tous les feux de signalisation. Ainsi, lorsqu'un feu est au vert, les autres sont au rouge." />
              <Paragraph text="Dans l'exemple suivant les feux de signalisations sont représentés par les traits de couleur rouge, vert et bleu." />
              <Image src={PIECE_OF_CAKE} />
            </Container>
          ),
          "Piece of Cake": (
            <Container
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="La stratégie 'Cross duplex' est la plus intuitive. Elle correspond au carrefour typique et permet de faire alterner les feux de signalisation des routes parallèles avec ceux des routes perpendiculaires." />
              <Paragraph text="Dans l'exemple suivant sont représentés en bleu et rouge les groupes de feux de signalisation. Le groupe bleu et le groupe rouge s'allument chacun leur tour. " />
              <Image src={CROSS_DUPLEX} />
            </Container>
          ),
        }}
      />
      <Title title="Développement et Implémentation" size="lg" />

      <Title title="Estimation Préliminaire des Performances" size="md" />
      <Paragraph text="Il existe trois points critiques pour la performance de notre projet:" />
      <AccordionParagraph
        children={{
          "🧪 Estimation du temps de génération des datasets": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="Pour générer 100,000 entrées (I, E) (on rappelle que le but est de 1,000,000 d'entrées):" />
              <Paragraph text="Il faut pouvoir exécuter 15 minutes de simulation dans un intervalle de temps réel de l'ordre de la seconde. Si on utilise une solution naive, il nous faudra ([nb_strategies] + 1) x [temps de simul 15 minutes] pour générer 1 entrée (I, E)." />
              <Paragraph text="Avec 10 stratégies, on a donc 11 secondes par entrée, soit 305 heures. Sur 8 coeurs cela revient à 38 heures de génération, ce qui est loin d'être négligeable." />
              <Paragraph text="En pratique:" />
              <Paragraph text="À notre grande surprise, nous arrivons à générer en python 15 minutes de simulation en 1.4 secondes, sans avoir fait d'optimisation. Nous pensons pouvoir descendre en dessous de la seconde." />
              <Paragraph text="Générer un dataset est encore très coûteux. Comme nous avons en réalité 15 stratégies, nous mettons environ 60 heures à générer 100000 entrées (I, E)" />
            </Box>
          ),
          "🎰 Temps d'entraînement": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="L'entraînement consiste à faire un forward + backward sur des batch générés par un dataloader. La complexité de cette tâche est de l'ordre de la taille du dataset utilisé. En pratique, l'entraînement d'un DNN à 64 neuronnes par couche sur un dataset de 100,000 entrées peut se fait en 5 minutes sans même utiliser de GPU (mesuré avec un exercice de pathfinding)." />
              <Paragraph text="En pratique:" />
              <Paragraph text="Pas de mauvaise surprise, nous arrivons à entraîner l'IA dans des ordres de grandeurs négligeables face au temps de génération des datasets." />
            </Box>
          ),
          "🏧 Coût de fonctionnement": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="L'IA en pratique serait utilisée une fois par croisement, toutes les 15 minutes. La performance n'est pas requise temporellement. Cependant, une meilleure performance permet de réduire la consommation énergétique, ce qui est un des buts principaux du projet. En effet, si l'énergie consommée n'était pas importante, il suffirait de lancer notre algorithme naif toutes les 15 minutes sur la simulation." />
              <Paragraph text="En pratique:" />
              <Paragraph text="Le coût temporel forward de l'IA est vastement négligeable face au coût temporel de la simulation (de l'ordre de 0.01%). Cela prouve que le coût énergétique d'un forward est lui aussi très faible." />
            </Box>
          ),
        }}
      />

      <ContinueLectureButton
        text="Continuer vers Simulation"
        href="/simulation"
      />
    </Container>
  );
};
