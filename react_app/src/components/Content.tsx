import { Container } from "@chakra-ui/layout";
import { theme } from "@chakra-ui/theme";
import { Banner } from "./Banner";
import { DocumentDescriptor } from "./DocumentDescriptor";
import { MultipleTabs } from "./MultipleTab";
import { StrategyCard } from "./StrategyCard";
import { CROSS_DUPLEX, OPEN_CORRIDOR, PIECE_OF_CAKE, OPEN } from "../assets";
import { Paragraph } from "./Paragraph";
import { ColorSwitch } from "./ColorSwitch";

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
      <Paragraph text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec euismod, nisl vitae aliquam ultricies, nunc nisl ultricies nunc, vitae aliquam nisl nisl vitae nisl. Donec euismod, nisl vitae aliquam ultricies, nunc nisl ultricies nunc, vitae aliquam nisl nisl vitae nisl." />
      <MultipleTabs
        childrens={{
          Open: <StrategyCard description="Open description" image={OPEN} />,
          "Open Corridor": (
            <StrategyCard
              description="Open Corridor description"
              image={OPEN_CORRIDOR}
            />
          ),
          "Cross Duplex": (
            <StrategyCard
              description="Cross Duplex description"
              image={CROSS_DUPLEX}
            />
          ),
          "Piece of Cake": (
            <StrategyCard
              description="Piece of Cake description"
              image={PIECE_OF_CAKE}
            />
          ),
        }}
      />
      <Paragraph text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec euismod, nisl vitae aliquam ultricies, nunc nisl ultricies nunc, vitae aliquam nisl nisl vitae nisl. Donec euismod, nisl vitae aliquam ultricies, nunc nisl ultricies nunc, vitae aliquam nisl nisl vitae nisl. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec euismod, nisl vitae aliquam ultricies, nunc nisl ultricies nunc, vitae aliquam nisl nisl vitae nisl. Donec euismod, nisl vitae aliquam ultricies, nunc nisl ultricies nunc, vitae aliquam nisl nisl vitae nisl." />
    </Container>
  );
};
