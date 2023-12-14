import {
  Table,
  TableCaption,
  TableContainer,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@chakra-ui/react";

export type ResultTabProps = {
  data: {
    date: string;
    time: string;
    description: string;
  }[];
  caption: string;
};

export const ResultTab = (props: ResultTabProps) => {
  return (
    <TableContainer>
      <Table variant="simple">
        <TableCaption>{props.caption}</TableCaption>
        <Thead>
          <Tr>
            <Th>Date</Th>
            <Th>Commentaire</Th>
            <Th isNumeric>Time (s)</Th>
          </Tr>
        </Thead>
        <Tbody>
          {props.data.map((value) => {
            return (
              <Tr>
                <Td>{value.date}</Td>
                <Td>{value.description}</Td>
                <Td isNumeric>{value.time}</Td>
              </Tr>
            );
          })}
        </Tbody>
      </Table>
    </TableContainer>
  );
};
