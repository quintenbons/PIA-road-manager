import { Container, theme } from "@chakra-ui/react";
import { BreadcrumbLnk } from "../components/BreadcrumbLnk";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Title } from "../components/Title";
import { Paragraph } from "../components/Paragraph";
import { ResultTab } from "../components/ResultTab";

export const Mesure = () => {
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
            name: "Mesure",
            href: "/mesure",
          },
        ]}
      />
      <DocumentDescriptor
        title="Mesures"
        date="13 Décembre, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
      />
      <Title title="Génération de datasets" size="md" />
      <Title title="Score" size="md" />
      <Title title="Performances Engine" size="md" />
      <ResultTab
        data={[
          {
            date: "14 Décembre, 2023",
            description: "",
            time: "1.2",
          },
        ]}
        caption={
          "Résultat de la mesure de performance de l'engine pour 15 minutes de jeu"
        }
      />
      <Title title="Performances Engine & Graphique" size="md" />
      <ResultTab
        data={[
          {
            date: "14 Décembre, 2023",
            description: "",
            time: "1.2",
          },
        ]}
        caption={
          "Résultat de la mesure de performance de l'engine avec l'interface graphique pour 15 minutes de jeu"
        }
      />
      <Title title="Pistes d'amélioration" size="md" />
      <Paragraph text="Nous n'avons pas réalisés de refactoring ou d'optimisation de la codebase. Nous pensons que cela pourrait largement améliorer les performances de l'engine." />
    </Container>
  );
};
