import { ChevronRightIcon } from "@chakra-ui/icons";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  theme,
} from "@chakra-ui/react";

export type BreadcrumbLnkProps = {
  sections: {
    name: string;
    href: string;
  }[];
  setPath: (path: string) => void;
};

export const BreadcrumbLnk = (props: BreadcrumbLnkProps) => {
  return (
    <Breadcrumb spacing="8px" separator={<ChevronRightIcon color="gray.500" />}>
      {props.sections.map((section, i) => (
        <BreadcrumbItem isCurrentPage={i === props.sections.length - 1}>
          <BreadcrumbLink
            color={theme.colors.green[400]}
            onClick={() => {
              window.scrollTo(0, 0);
              props.setPath(section.href);
            }}
          >
            {section.name}
          </BreadcrumbLink>
        </BreadcrumbItem>
      ))}
    </Breadcrumb>
  );
};
