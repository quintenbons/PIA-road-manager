import { ChakraProvider } from "@chakra-ui/react";
import { NavBar } from "./components/NavBar";
import { Content } from "./components/Content";

function App() {
  return (
    <ChakraProvider>
      <NavBar />
      <Content />
    </ChakraProvider>
  );
}

export default App;
