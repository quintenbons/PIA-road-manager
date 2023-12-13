import { Container, Heading, Text } from "@chakra-ui/layout";
import { theme } from "@chakra-ui/theme";
import { Banner } from "./Banner";
import { DocumentDescriptor } from "./DocumentDescriptor";

export const Content = () => {
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
        title="Rapport IA"
        date="13 Décembre, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
      />
    </Container>
  );
};
