import { Container, theme } from "@chakra-ui/react";
import { BreadcrumbLnk } from "../components/BreadcrumbLnk";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Title } from "../components/Title";
import { AllLinks } from "../components/AllLinks";

export const References = () => {
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
            name: "Références",
            href: "/references",
          },
        ]}
      />
      <DocumentDescriptor
        title="Références"
        date="13 Décembre, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
      />
      <Title title="Références" size="md" />

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
          {
            text: "Pygame",
            url: "https://www.pygame.org/news",
          },
        ]}
      />
    </Container>
  );
};
