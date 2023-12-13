import { Box, Container, Icon, Text, theme } from "@chakra-ui/react";

export const NavBar = () => {
  return (
    <Box
      sx={{
        width: "100%",
        backgroundColor: theme.colors.green[100],
      }}
    >
      <Container
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-start",
          height: theme.sizes[16],
          gap: theme.space[4],
          padding: theme.space[4],
        }}
      >
        <Icon w={8} h={8}>
          Icon
        </Icon>
        <Text
          sx={{
            fontSize: theme.fontSizes["lg"],
            fontWeight: theme.fontWeights.semibold,
            verticalAlign: "middle",
            textAlign: "center",
          }}
        >
          Text
        </Text>
      </Container>
    </Box>
  );
};
