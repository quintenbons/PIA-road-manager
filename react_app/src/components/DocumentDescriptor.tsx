import { Box, Heading, Text } from "@chakra-ui/layout";
import { theme } from "@chakra-ui/theme";

export type DocumentDescriptorProps = {
  title: string;
  date: string;
  authors: string[];
};

export const DocumentDescriptor = (props: DocumentDescriptorProps) => {
  return (
    <Box sx={{
        display: "flex",
        flexDirection: "column",
        gap: theme.space[1],
    }}>
      <Text
        sx={{
          color: theme.colors.gray[500],
        }}
      >
        📅 {props.date}
      </Text>
      <Heading>{props.title}</Heading>
      <Text
        sx={{
          color: theme.colors.gray[500],
        }}
      >
        ✍️ {props.authors.join(", ")}
      </Text>
    </Box>
  );
};
