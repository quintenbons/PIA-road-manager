import { ArrowForwardIcon } from "@chakra-ui/icons";
import { Box, Button } from "@chakra-ui/react";

export type ContinueLectureButtonProps = {
  href: string;
  text: string;
  setPath: (path: string) => void;
};

export const ContinueLectureButton = (props: ContinueLectureButtonProps) => {
  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        width: "100%",
      }}
    >
      <Button
        colorScheme="green"
        variant="outline"
        rightIcon={<ArrowForwardIcon />}
        onClick={() => {
          window.scrollTo(0, 0);
          props.setPath(props.href);
        }}
      >
        {props.text}
      </Button>
    </Box>
  );
};
