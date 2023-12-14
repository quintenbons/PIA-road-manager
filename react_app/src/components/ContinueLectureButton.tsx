import { ArrowForwardIcon } from "@chakra-ui/icons";
import { Box, Button } from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";

export type ContinueLectureButtonProps = {
  href: string;
  text: string;
};

export const ContinueLectureButton = (props: ContinueLectureButtonProps) => {
  const navigate = useNavigate();
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
          navigate(props.href);
        }}
      >
        {props.text}
      </Button>
    </Box>
  );
};
