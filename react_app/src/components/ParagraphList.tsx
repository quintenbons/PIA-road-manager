import { ListItem, UnorderedList } from "@chakra-ui/layout";

export type ParagraphListProps = {
  paragraphs: string[];
};

export const ParagraphList = (props: ParagraphListProps) => {
  return (
    <UnorderedList>
      {props.paragraphs.map((paragraph) => {
        return <ListItem>{paragraph}</ListItem>;
      })}
    </UnorderedList>
  );
};
