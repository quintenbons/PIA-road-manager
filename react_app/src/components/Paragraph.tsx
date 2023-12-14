import { Text } from "@chakra-ui/layout";

export type ParagraphProps = {
  text: string;
};

export const Paragraph = (props: ParagraphProps) => {
  return (
    <Text
      sx={{
        textAlign: "justify",
      }}
    >
      {props.text}
    </Text>
  );
};
