import { Box, Container, Image, Text } from "@chakra-ui/react";

export type StrategyCardProps = {
  description: string;
  image: string;
};

export const StrategyCard = (props: StrategyCardProps) => {
  return (
    <Container>
      <Text>{props.description}</Text>
      <Image src={props.image} />
    </Container>
  );
};
