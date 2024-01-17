import { ExternalLinkIcon } from "@chakra-ui/icons";
import { Link } from "@chakra-ui/layout";
import {
  Table,
  TableContainer,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@chakra-ui/table";

export type GlossaryProps = {
  words: { text: string; definition: string }[];
};

export const Glossary = (props: GlossaryProps) => {
  return (
    <TableContainer>
      <Table variant="simple">
        <Thead>
          <Tr>
            <Th>Mot</Th>
            <Th>Definition</Th>
          </Tr>
        </Thead>
        <Tbody>
          {props.words.map((word) => {
            return (
              <Tr>
                <Td>{word.text}</Td>
                <Td>{word.definition}</Td>
              </Tr>
            );
          })}
        </Tbody>
      </Table>
    </TableContainer>
  );
};
