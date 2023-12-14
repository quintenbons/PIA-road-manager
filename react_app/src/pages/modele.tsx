import { Container, theme, Image } from "@chakra-ui/react";
import { BreadcrumbLnk } from "../components/BreadcrumbLnk";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Title } from "../components/Title";
import { Paragraph } from "../components/Paragraph";
import { ParagraphList } from "../components/ParagraphList";
import { GRENOBLE } from "../assets";

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
      <Title title="Modèle d'IA : Dense Neural Network" size="md" />
      <Paragraph text="Afin de prendre en main Pytorch, un court exercice a été fait : entraîner un modèle de Pathfinding. Cela nous a aussi permis de faire des premières estimations de performance (@perf)." />
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
          "Branchement à des cas concrêts (transformation d'une situation réelle en entrée du dataset)",
        ]}
      />
    </Container>
  );
};
