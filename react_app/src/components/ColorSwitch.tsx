import { useColorMode } from "@chakra-ui/color-mode";
import { Icon } from "@chakra-ui/icon";
import { Box } from "@chakra-ui/layout";
import { Switch } from "@chakra-ui/switch";
import { MoonIcon, SunIcon } from "@chakra-ui/icons";
import { theme } from "@chakra-ui/theme";

export const ColorSwitch = () => {
  const { toggleColorMode, colorMode } = useColorMode();
  console.log(colorMode);
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "flex-end",
        gap: theme.space[4],
        padding: `0 ${theme.space[4]} 0 ${theme.space[4]}`,
      }}
    >
      <Icon as={SunIcon} boxSize={"1.5rem"} />
      <Switch
        colorScheme="teal"
        size="lg"
        onChange={toggleColorMode}
        isChecked={colorMode === "dark"}
      />
      <Icon as={MoonIcon} boxSize={"1.5rem"} />
    </Box>
  );
};
