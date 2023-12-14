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
      <Paragraph text="Jusqu'à présent nous avons dégagé un total de cinq stratégies primaires. Ces stratégies représentent les logiques s'appliquant aux feux. Les règles de circulation classiques telles que les priorités à droite ou les distances de sécurité s'appliquent en priorité sur les stratégies." />
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
      <Title title="Entraînement du Modèle" size="lg" />
      <Title title="Format des Données d'entraînement" size="md" />
      <Paragraph
        text="Pour entraîner efficacement notre modèle d'intelligence artificielle basé sur PyTorch, il est essentiel d'avoir des datasets structurés pour une utilisation avec des dataloaders. Ces datasets permettront de nourrir et d'affiner le modèle d'IA, en fournissant une source de données cohérente et structurée, adaptée aux spécificités de l'apprentissage automatique.
    Nous estimons qu'un volume conséquent, de plusieurs millions d'exemples au moins, sera nécessaire pour atteindre une efficacité optimale."
      />
      <Paragraph text="Le dataset sera constitué de tuples (I, E)." />
      <ParagraphList
        paragraphs={[
          "I (input) correspond à une entrée du réseau neuronal pour une intersection donnée. I contient aussi des informations sur les évènements des 15 minutes précédentes.",
          "E (expected) correspond au résultat attendu 'expected', qui devra être généré algorithmiquement.",
        ]}
      />
      <Paragraph text="Cette approche structurée permettra à notre modèle de s'adapter et d'apprendre efficacement à partir d'un large éventail de scénarios de trafic." />
      <Title title="Génération des données d'entrée" size="md" />
      <Image src={DATASET_GENERATION} width={"60%"} alignSelf={"center"} />

      <Paragraph text="La génération de jeux de données pour entraîner notre IA est un processus clé, impliquant la création de configurations spécifiques pour simuler divers scénarios de trafic urbain. Il est important de rappeler que notre modèle n'est conscient que d'une seule intersection, ce qui nous permet de simplifier drastiquement la génération des datasets, en limiant la simulation à un seul noeud." />
      <Paragraph text="Pour générer une entrée de dataset (I, E), la simulation est d'abord exécutée pendant 15 minutes avec une stratégie sélectionnée uniformément, sur une configuration aléatoire. À l'issue de cette période, le paramètre I de l'entrée de dataset peut être calculé à partir des informations mesurées pendant l'exécution." />
      <Paragraph text="Il faut ensuite générer le paramètre E, qui correspond au meilleur choix possible de stratégie pour les 15 minutes suivantes. Pour cela, nous pouvons simplement exécuter les stratégies une par une, et récupérer celle qui obtient le meilleur score (le moins de congestion). Attention ici à bien utiliser la même configuration de trafic que pour les 15 minutes initiales pour ne pas enfreindre notre hypothèse de consistence du trafic." />

      <Image src={DATASET_HESITATION} width={"60%"} alignSelf={"center"} />
      <Paragraph text="Un cas particulier est à prévoir (voir @close-second) : si plusieurs stratégies sont correctes, pénaliser le modèle lors du backwards peut être néfaste. C'est pourquoi nous détecterons ces situations afin de les exclure du dataset." />
      <Paragraph text="Il est possible à l'avenir que nous reviendrons sur cette décision afin de permettre à l'intelligence artificielle d'apprendre à résoudre des dilemmes plus efficacement." />
      <Title title="Développement et Implémentation" size="lg" />
  
      <Title title="Estimation Préliminaire des Performances" size="md" />
      <Paragraph text="Il existe trois points critiques pour la performance de notre projet:" />
      <AccordionParagraph
        children={{
          "🧪 Estimation": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="Pour générer 100,000 entrées (I, E) (on rappelle que le but est de 1,000,000 d'entrées):" />
              <Paragraph text="Il faut pouvoir exécuter 15 minutes de simulation dans un intervalle de temps réel de l'ordre de la seconde. Si on utilise une solution naive, il nous faudra ([nb_strategies] + 1) x [temps de simul 15 minutes] pour générer 1 entrée (I, E). (Cf @close-second)" />
              <Paragraph text="Avec 10 stratégies, on a donc 11 secondes par entrée, soit 305 heures. Sur 8 coeurs cela revient à 38 heures de génération, ce qui est loin d'être négligeable." />
              <Paragraph text="En pratique:" />
              <Paragraph text="Le simulateur est écrit en python, et gère des intervalles de temps de l'ordre de la seconde. Il probable que l'objectif de 1 seconde pour 15 minutes de simulation est actuellement impossible. C'est pour cela que nous devons porter une grande importance à la performance de la simulation." />
            </Box>
          ),
          "🎰 Génération de dataset": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="L'entraînement consiste à faire un forward + backward sur des batch générés par un dataloader. La complexité de cette tâche est de l'ordre de la taille du dataset utilisé. En pratique, l'entraînement d'un DNN à 64 neuronnes par couche sur un dataset de 100,000 entrées peut se fait en 5 minutes sans même utiliser de GPU (mesuré avec un exercice de pathfinding)." />
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
            </Box>
          ),
        }}
      />

      <Title title="Conclusion" size="lg" />
      <Paragraph text="Le développement du modèle et de la simulation ont avancé en parallèle. Les deux blocs de ce projet sont donc dans un état avancé. Cependant, la chaîne de dépendance des tâches nous oblige à encore patienter avant de pouvoir brancher les deux parties." />
      <Paragraph text="Il est important à présent de se concentrer sur la génération de datasets afin d'avoir d'afiner nos estimations, et éventuellement prendre des mesures pour rendre la génération plus rapide." />
      <Title title="Réferences" size="md" />

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
        ]}
      />
    </Container>
  );
};
