import { Container, theme, Image, Code } from "@chakra-ui/react";
import { BreadcrumbLnk } from "../components/BreadcrumbLnk";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Title } from "../components/Title";
import { Paragraph } from "../components/Paragraph";
import { ParagraphList } from "../components/ParagraphList";
import { ContinueLectureButton } from "../components/ContinueLectureButton";
import { DATASET_GENERATION, DATASET_HESITATION } from "../assets";

export const Modele = () => {
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
        date="13 Décembre, 2023"
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
      <Paragraph text="Un cas particulier est à prévoir (voir schéma ci-dessus) : si plusieurs stratégies sont correctes, pénaliser le modèle lors du backwards peut être néfaste. C'est pourquoi nous détecterons ces situations afin de les exclure du dataset." />
      <Paragraph text="Nous hésitons aussi à passer d'un modèle one-hot à un modèle softmax probabiliste. C'est quelque chose que nous avions déjà fait remarquer lors du dernier livrable, et qui n'est toujours pas sûr." />
      <Paragraph text="Note: les scores sont en réalité de l'ordre de plusieurs millions. Plus d'information dans la page 'Mesures'." />

      <Title title="Modèle d'IA : Dense Neural Network" size="md" />
      <Paragraph text="Afin de prendre en main Pytorch, un court exercice a été fait : entraîner un modèle de Pathfinding. Cela nous a aussi permis de faire des premières estimations de performance (sur la page Home)." />
      <Paragraph text="Une infrastructure d'entraînement pour le projet routier est déjà prête. Elle contient" />
      <ParagraphList
        paragraphs={[
          "Un module de manipulation des datasets",
          "Un module d'entraînement, prennant en entrée un dataloader, et le modèle",
          "Possibilité de sauvegarder les datasets et modèles",
        ]}
      />
      <Paragraph text="Il lui manque:" />
      <ParagraphList
        paragraphs={[
          "Des fonctionalités de génération automatique de dataset à partir de configs externes",
          "Branchement à des cas concrets (transformation d'une situation réelle en entrée du dataset)",
        ]}
      />

      <Code colorScheme="green">
        class CrossRoadModel(nn.Module):
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;def __init__(self, nb_strats=OUTPUT_DIM):
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;super(CrossRoadModel,
        self).__init__()
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.fc1 =
        nn.Linear(INPUT_DIM, HIDDEN_DIM)
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.fc2 =
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.fc3 =
        nn.Linear(HIDDEN_DIM, nb_strats)
        <br />
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;def forward(self, x):
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = F.relu(self.fc1(x))
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = F.relu(self.fc2(x))
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = F.relu(self.fc3(x))
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return x<br />
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;def save(self, target: PathLike):
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if isinstance(self,
        nn.DataParallel):
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;state_dict
        = self.module.state_dict()
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else:
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;state_dict
        = self.state_dict()
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;torch.save(state_dict,
        target)
        <br />
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;@classmethod
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;def load(Cls, target: PathLike, device="cpu"):
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;state_dict =
        torch.load(target, map_location=device)
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model = Cls()
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model.load_state_dict(state_dict)
        <br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return model
        <br />
      </Code>

      <ContinueLectureButton text="Continuer vers Mesures" href="/mesures" />
    </Container>
  );
};
