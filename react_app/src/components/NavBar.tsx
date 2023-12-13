import {
  Box,
  Icon,
  theme,
  useColorMode,
} from "@chakra-ui/react";
import { ColorSwitch } from "./ColorSwitch";

export const NavBar = () => {
  const { colorMode } = useColorMode();
  return (
    <Box
      sx={{
        width: "100%",
        backgroundColor:
          colorMode === "light"
            ? theme.colors.green[100]
            : theme.colors.green[900],
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
      {/* <Text
          sx={{
            fontSize: theme.fontSizes["lg"],
            fontWeight: theme.fontWeights.semibold,
            verticalAlign: "middle",
            textAlign: "center",
          }}
        >
          Text
        </Text> */}
      <Box
        sx={{
          marginLeft: "auto",
        }}
      >
        <ColorSwitch />
      </Box>
    </Box>
  );
};
