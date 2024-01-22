import { Container, theme, Image, Code, Box } from "@chakra-ui/react";
import { BreadcrumbLnk } from "../components/BreadcrumbLnk";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Title } from "../components/Title";
import { Paragraph } from "../components/Paragraph";
import { ParagraphList } from "../components/ParagraphList";
import { ContinueLectureButton } from "../components/ContinueLectureButton";
import {
  ANGLE_ORDERING,
  CURRENT_MODEL,
  DATASET_GENERATION,
  DATASET_HESITATION,
  ORDER_DIFF,
  TBOTTOM,
  TLEFT,
  TRIGHT,
  TTOP,
} from "../assets";
import LineChart from "../components/LineChart";
import training_data from "../data/first_training_data.json";
import large_training_data from "../data/large_training_data.json";

export const Modele = (props: { setPath: (path: string) => void }) => {
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
            name: "Modèle",
            href: "/modele",
          },
        ]}
      />
      <DocumentDescriptor
        title="Modèle"
        date="22 Janvier, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
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
          "I (input) correspond à une entrée du réseau neuronal pour une intersection donnée. Chaque route collecte pendant 15 minutes 4 mesures de trafic: nombre de véhicules (entrants/sortants), nombre de secondes passées au feu rouge (première voiture/cumulé). Ces mesures sont ensuite normalisées et concaténées pour former le vecteur d'entrée I.",
          "E (expected) correspond au résultat attendu 'expected', qui devra être généré algorithmiquement.",
        ]}
      />
      <Paragraph text="Cette approche structurée permettra à notre modèle de s'adapter et d'apprendre efficacement à partir d'un large éventail de scénarios de trafic." />
      

      <Title title="Génération des données d'entrée" size="md" />
      
      <Paragraph text="La génération de jeux de données pour entraîner notre IA est un processus clé, impliquant la création de configurations spécifiques pour simuler divers scénarios de trafic urbain. Il est important de rappeler que notre modèle n'est conscient que d'une seule intersection, ce qui nous permet de simplifier drastiquement la génération des datasets, en limitant la simulation à un seul noeud." />
      
      <Title title="Génération des cartes" size="sm" />
      
      <Paragraph text="L'IA est entraînée sur des cartes créées à la main. Nous l'avons entrainé sur les intersections à 3, 4 et 5 routes entrantes (et sortantes)." />      
      <Paragraph text="Après quelques expérimentations, nous avons réalisé que l'IA était sensible l'ordre des routes d'une intersection. Par exemple, pour les intersections à trois branches en forme de T, nous avons dû entrainer l'IA sur ces différentes cartes:" />
      
      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "center",
          alignItems: "center",
          gap: theme.space[4],
        }}
      >
        <Image
          src={TLEFT}
          flex={"1"}
          width={"25%"}
          alignSelf={"center"}
        />
        <Image src={TRIGHT} flex={"1"} width={"25%"} alignSelf={"center"} />
        <Image src={TTOP} flex={"1"} width={"25%"} alignSelf={"center"} />
        <Image src={TBOTTOM} flex={"1"} width={"25%"} alignSelf={"center"} />
      </Box>

      <Paragraph text="Mais ce n'est pas assez, car pour l'IA, ces deux configurations sont différentes:" />
      <Image src={ORDER_DIFF} width={"60%"} alignSelf={"center"} />
      <Paragraph text="Ici, la numérotation des sommets se fait dans un ordre arbitraire qui dépend de la création de la carte. Pour la même forme on peut avoir une numérotation différente et donc des interprétations de l'IA différente. Pour pallier cette problématique, nous effectuons un réordonnancement dynamique de la numérotation des sommets avant chaque simulation par rapport à l'axe (1,0) ." />
      <Image src={ANGLE_ORDERING} height={"400px"} alignSelf={"center"} />
      
      <Paragraph text="Ainsi et grâce à ce réordonnancement, il ne reste que trois ordres de sommets possibles et on élimine une des quatre cartes d'entrainement." />

      <Title title="Génération des labels" size="sm" />

      <Paragraph text="Pour générer une entrée de dataset (I, E), la simulation est d'abord exécutée pendant 15 minutes avec une stratégie sélectionnée uniformément, sur une configuration aléatoire. À l'issue de cette période, le paramètre I de l'entrée de dataset peut être calculé à partir des informations mesurées pendant l'exécution." />
      <Image src={DATASET_GENERATION} width={"60%"} alignSelf={"center"} />
      <Paragraph text="Il faut ensuite générer le paramètre E, qui correspond au label de l'entrée de dataset. Dans notre cas, nous avons choisi de ne pas demander un one-hot du meilleur résultat (B), mais plutôt une estimation des scores en softmax inverse. Pour cela, nous pouvons simplement exécuter les stratégies une par une, et récupérer celle qui obtient le meilleur score (le moins de congestion). Attention ici à bien utiliser la même configuration de trafic que pour les 15 minutes initiales pour ne pas enfreindre notre hypothèse de consistence du trafic. (La seed est aussi fixée pour ne pas faire de jaloux)" />

      <Image src={DATASET_HESITATION} width={"60%"} alignSelf={"center"} />
      <Paragraph text="Le cas particulier ci dessus est à prévoir : si plusieurs stratégies sont correctes, pénaliser le modèle lors du backwards peut être néfaste. Le fait d'attendre une estimation des scores permet de ne pas trop punir l'IA dans ce genre de situations." />
      <Paragraph text="Note: Nous avions initialement choisi d'utiliser un label one-hot, qui donnait des résultats bien pires que ceux que nous avons maintenant." />
      <Paragraph text="Note: les scores sont en réalité de l'ordre de plusieurs millions. Plus d'information dans la page 'Mesures'." />

      <Title title="Modèle d'IA : Dense Neural Network" size="md" />
      <Paragraph text="Afin de prendre en main Pytorch, un court exercice a été fait : entraîner un modèle de Pathfinding. Cela nous a aussi permis de faire des premières estimations de performance (sur la page Home)." />
      <Paragraph text="Une infrastructure d'entraînement pour le projet routier est disponible. Nous y avons mis beaucoup d'effort afin de rentre le projet assez générique pour pouvoir essayer de nouvelles structures de modèle et d'entrées par exemple. Nous vous conseillons de regarder le README du projet si vous souhaitez en voir la complétude. Elle contient:" />
      <ParagraphList
        paragraphs={[
          "Manipulation des datasets (suppression, split, merge, lecture, description détaillée)",
          "Entraînement, prennant en entrée un dataloader, et le modèle",
          "Sauvegarde des datasets et modèles",
          "Génération automatique de dataset à partir de configs externes",
          "Génération à distance (cluster pré-déployé avec des clés SSH) dans notre cas nous nous sommes permis d'utiliser des vmgpu de l'ensimag pendant 2 heures.",
          "Branchement à des cas concrets (transformation d'une situation réelle en entrée du dataset avec une fonction torchify)",
        ]}
      />

      <Paragraph text="Code de notre modèle:" />

      <Code colorScheme="green">
        class CrossRoadModel(nn.Module):
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;inp: nn.Linear
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;hidden_layers: nn.ModuleList
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;out: nn.Linear
        <br />
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;input_size: int = INPUT_DIM
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;output_size: int = OUTPUT_DIM
        <br />
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, nb_strats=OUTPUT_DIM):
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;super(CrossRoadModel, self).__init__()
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.inp = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.hidden_layers = nn.ModuleList([nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(HIDDEN_AMOUNT)])
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.out = nn.Linear(HIDDEN_DIM, nb_strats)
        <br />
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.input_size = INPUT_DIM
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.output_size = nb_strats
        <br />
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;def forward(self, x):
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = F.relu(self.inp(x))
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for layer in self.hidden_layers:
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = F.relu(layer(x))
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = F.relu(self.out(x))
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return x
      </Code>

      <Paragraph text="Schéma représentatif:" />

      <Image src={CURRENT_MODEL} height={"30rem"} alignSelf={"center"} />

      <Title title="Entraînement du modèle sur 400 entrées" size="md" />
      <LineChart data={training_data} entryNumber={400} minRange={2} />
      <Paragraph text="Ce graphe date du dernier rapport. Nous n'avions pas encore d'entrée conséquente, et le label était en one-hot. Nous nous attendions à une divergeance des résultats pour quelques raisons: dataset trop petite, trop peu d'input, modèle trop peu profond (nous commenterons cela dans la section exploration)." />

      <Title title="Entraînement du modèle sur 189k entrées" size="md" />
      <LineChart data={large_training_data} entryNumber={189000} minRange={2.5} />
      <Paragraph text="Comme vous pouvez le constater, L'entraînement ne fournit pas encore des courbes de pertes visuellement satisfaisantes. Cependant nous en avons compris les raisons, et avons su tirer profit de cette expérience (nous le verrons dans la section exploration)." />

      <ContinueLectureButton
        text="Continuer vers Mesures"
        href="/mesures"
        setPath={props.setPath}
      />
    </Container>
  );
};
