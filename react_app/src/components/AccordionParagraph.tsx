import {
  Accordion,
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
} from "@chakra-ui/accordion";
import { useColorMode } from "@chakra-ui/color-mode";
import { Box } from "@chakra-ui/layout";
import { theme } from "@chakra-ui/theme";

export type AccordionParagraphProps = {
  children: { [x: string]: React.ReactNode };
};

export const AccordionParagraph = (props: AccordionParagraphProps) => {
  const { colorMode } = useColorMode();
  return (
    <Accordion allowMultiple>
      {Object.keys(props.children).map((key) => {
        return (
          <AccordionItem>
            <h2>
              <AccordionButton
                sx={{
                  _hover: {
                    color: theme.colors.green[400],
                    borderColor: theme.colors.green[400],
                    borderBottomWidth: 0,
                    backgroundColor:
                      colorMode === "light"
                        ? theme.colors.gray[50]
                        : theme.colors.gray[700],
                  },
                }}
              >
                <Box as="span" flex="1" textAlign="left">
                  {key}
                </Box>
                <AccordionIcon />
              </AccordionButton>
            </h2>
            <AccordionPanel pb={4}>{props.children[key]}</AccordionPanel>
          </AccordionItem>
        );
      })}
    </Accordion>
  );
};
