import { ChakraProvider } from "@chakra-ui/react";
import { NavBar } from "./components/NavBar";
import { Home } from "./pages/home";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Mesures } from "./pages/mesures";
import { Simulation } from "./pages/simulation";
import { Modele } from "./pages/modele";
import { Footer } from "./components/Footer";
import { References } from "./pages/references";
import { useNavigation } from "./navigation";
import { Exploration } from "./pages/Exploration";

function App() {
  const { path, setPath } = useNavigation();
  return (
    <BrowserRouter>
      <ChakraProvider>
        <NavBar
          setPath={setPath}
          sections={[
            {
              name: "Home",
              href: "/",
            },
            {
              name: "Simulation",
              href: "/simulation",
            },
            {
              name: "Modèle",
              href: "/modele",
            },
            {
              name: "Mesures",
              href: "/mesures",
            },
            {
              name: "Exploration",
              href: "/exploration",
            },
            {
              name: "Références",
              href: "/references",
            },
          ]}
        />
        <>
          {path === "/" ? (
            <Home setPath={setPath} />
          ) : path === "/simulation" ? (
            <Simulation setPath={setPath} />
          ) : path === "/modele" ? (
            <Modele setPath={setPath} />
          ) : path === "/mesures" ? (
            <Mesures setPath={setPath} />
          ) : path === "/exploration" ? (
            <Exploration setPath = {setPath} />
          ) : path === "/references" ? (
            <References setPath={setPath} />
          ) : (
            <Home setPath={setPath} />
          )}
        </>
        <Footer />
      </ChakraProvider>
    </BrowserRouter>
  );
}

export default App;
