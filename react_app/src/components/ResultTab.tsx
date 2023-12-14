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
    commitHash: string;
    granularity: string;
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
            <Th>Commit hash</Th>
            <Th isNumeric>Temps (s)</Th>
            <Th isNumeric>Granularité (s)</Th>
          </Tr>
        </Thead>
        <Tbody>
          {props.data.map((value) => {
            return (
              <Tr>
                <Td>{value.date}</Td>
                <Td>{value.commitHash}</Td>
                <Td isNumeric>{value.time}</Td>
                <Td isNumeric>{value.granularity}</Td>
              </Tr>
            );
          })}
        </Tbody>
      </Table>
    </TableContainer>
  );
};
