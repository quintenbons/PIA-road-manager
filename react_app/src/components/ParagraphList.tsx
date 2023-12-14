import { ListItem, UnorderedList, Text } from "@chakra-ui/layout";

export type ParagraphListProps = {
  paragraphs: string[];
};

export const ParagraphList = (props: ParagraphListProps) => {
  return (
    <UnorderedList>
      {props.paragraphs.map((paragraph) => {
        return (
          <ListItem>
            <Text
              sx={{
                textAlign: "justify",
              }}
            >
              {paragraph}
            </Text>
          </ListItem>
        );
      })}
    </UnorderedList>
  );
};
