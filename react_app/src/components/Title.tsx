import { Heading } from "@chakra-ui/layout";

export type TitleProps = {
  title: string;
  size?: string;
};

export const Title = (props: TitleProps) => {
  return <Heading size={props.size}>{props.title}</Heading>;
};
