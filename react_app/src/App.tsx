import { ChakraProvider } from "@chakra-ui/react";
import { NavBar } from "./components/NavBar";
import { Home } from "./pages/home";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Mesures } from "./pages/mesures";
import { Simulation } from "./pages/simulation";
import { Modele } from "./pages/modele";
import { Footer } from "./components/Footer";
import { References } from "./pages/references";

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
              name: "Mesures",
              href: "/mesures",
            },
            {
              name: "Références",
              href: "/references",
            },
          ]}
        />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="simulation" element={<Simulation />} />
          <Route path="mesures" element={<Mesures />} />
          <Route path="modele" element={<Modele />} />
          <Route path="references" element={<References />} />
        </Routes>
        <Footer />
      </ChakraProvider>
    </BrowserRouter>
  );
}

export default App;
