import { ChakraProvider } from "@chakra-ui/react";
import { NavBar } from "./components/NavBar";
import { Home } from "./pages/home";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Mesure } from "./pages/mesure";
import { Simulation } from "./pages/simulation";
import { Modele } from "./pages/modele";
import { Footer } from "./components/Footer";

function App() {
  return (
    <BrowserRouter>
      <ChakraProvider>
        <NavBar
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
              name: "Mesure",
              href: "/mesure",
            },
          ]}
        />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="simulation" element={<Simulation />} />
          <Route path="mesure" element={<Mesure />} />
          <Route path="modele" element={<Modele />} />
        </Routes>
        <Footer />
      </ChakraProvider>
    </BrowserRouter>
  );
}

export default App;
