import {
  Link,
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
    comment: string;
  }[];
  caption: string;
  githubLink: string;
};

function minimizeCommitHash(hash: string) {
  // d9a51775c2bd61c1a0e4e87a2ba83d9de7b6c463 to d9a...463
  return hash.slice(0, 3) + "..." + hash.slice(-3);
}

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
            <Th>Commentaire</Th>
          </Tr>
        </Thead>
        <Tbody>
          {props.data.map((value) => {
            return (
              <Tr>
                <Td>{value.date}</Td>
                <Td>
                  <Link target="_blank" href={props.githubLink + "/commit/" + value.commitHash}>
                    {minimizeCommitHash(value.commitHash)}
                  </Link>
                </Td>
                <Td isNumeric>{Number(value.time).toFixed(3)}</Td>
                <Td>{value.comment}</Td>
              </Tr>
            );
          })}
        </Tbody>
      </Table>
    </TableContainer>
  );
};
