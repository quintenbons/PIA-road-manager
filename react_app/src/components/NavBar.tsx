import { Box, Icon, theme, Text, useColorMode, Link } from "@chakra-ui/react";
import { ColorSwitch } from "./ColorSwitch";
import { FaGithub } from "react-icons/fa";

export type NavBarProps = {
  sections: {
    name: string;
    href: string;
  }[];
  setPath: (path: string) => void;
};

export const NavBar = (props: NavBarProps) => {
  const { colorMode } = useColorMode();
  return (
    <>
      <Box
        sx={{
          zIndex: 1,
          width: "100vw",
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
          position: "fixed",
        }}
      >
        <a href="https://github.com/quintenbons/PIA-road-manager">
          <Icon as={FaGithub} w={8} h={8}>
            Icon
          </Icon>
        </a>
        {props.sections.map((section) => (
          <Link
            sx={{
              verticalAlign: "middle",
              textAlign: "center",
            }}
            onClick={() => {
              window.scrollTo(0, 0);
              props.setPath(section.href);
            }}
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
      <Box
        sx={{
          width: "100vw",
          height: theme.sizes[16],
          padding: theme.space[4],
        }}
      />
    </>
  );
};
