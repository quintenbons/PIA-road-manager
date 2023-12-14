import { Box, Icon, theme, Text, useColorMode, Link } from "@chakra-ui/react";
import { ColorSwitch } from "./ColorSwitch";

export type NavBarProps = {
  sections: {
    name: string;
    href: string;
  }[];
};

export const NavBar = (props: NavBarProps) => {
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
        gap: theme.space[7],
        padding: theme.space[4],
      }}
    >
      <Icon w={8} h={8}>
        Icon
      </Icon>
      {props.sections.map((section) => (
        <Link
          sx={{
            verticalAlign: "middle",
            textAlign: "center",
          }}
          href={section.href}
        >
          {section.name}
        </Link>
      ))}
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
