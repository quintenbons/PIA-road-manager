import { ExternalLinkIcon } from "@chakra-ui/icons";
import { Link } from "@chakra-ui/layout";
import {
  Table,
  TableCaption,
  TableContainer,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@chakra-ui/table";

export type AllLinksProps = {
  links: { text: string; url: string }[];
};

export const AllLinks = (props: AllLinksProps) => {
  return (
    <TableContainer>
      <Table variant="simple">
        <TableCaption>Imperial to metric conversion factors</TableCaption>
        <Thead>
          <Tr>
            <Th>Source</Th>
          </Tr>
        </Thead>
        <Tbody>
          {props.links.map((link) => {
            return (
              <Tr>
                <Td>
                  <Link href={link.url} isExternal>
                    {link.text} <ExternalLinkIcon mx="2px" />
                  </Link>
                </Td>
              </Tr>
            );
          })}
        </Tbody>
      </Table>
    </TableContainer>
  );
};
