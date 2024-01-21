import { Container, theme, Image } from "@chakra-ui/react";
import { BreadcrumbLnk } from "../components/BreadcrumbLnk";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Title } from "../components/Title";
import { Paragraph } from "../components/Paragraph";
import { DEEP_SPIKES, SHAPECMP } from "../assets";

export const Exploration = (props: { setPath: (path: string) => void }) => {
  return (
    <Container
      maxW="container.lg"
      sx={{
        padding: theme.space[4],
        gap: theme.space[4],
        display: "flex",
        flexDirection: "column",
      }}
    >
      <BreadcrumbLnk
        setPath={props.setPath}
        sections={[
          {
            name: "Home",
            href: "/",
          },
          {
            name: "Références",
            href: "/references",
          },
        ]}
      />
      <DocumentDescriptor
        title="Exploration"
        date="13 Décembre, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
      />
      <Title title="But" size="md" />
      <Paragraph text="Cette courte page conclut notre rapport. Le but ici est de souligner les différentes pistes que nous aurions aimé explorer plus, ou que nous avons pu explorer sans succès jusqu'à présent." />

      <Title title="Modification du Simulateur: troquer réalisme contre efficacité de l'IA" size="md" />
      <Paragraph text="Les conséquences des choix des stratégies ne sont pas assez importantes actuellement. Comme vous avez pu le voir sur les courbes de benchmarks, nos scores sont souvent très proches, et nos stratégies très similauires." />
      <Paragraph text="Nous sommes néanmoins fier de notre innovation sur les stratégies existantes et sur leur degré de généricité, mais nous pensons qu'appliquer un modèle d'une telle façon sur un problème qui se veut réaliste n'est pas facile." />
      <Paragraph text="Nous aurions aimé explorer plus une simulation peut être moins réaliste, mais qui accorde plus d'importance au choix d'une stratégie pour pouvoir mieux ressentir l'impact de l'IA. C'est ce que nous avons commencé à faire sur la branche main. Si vous en avez la motivation, vous pouvez essayer par vous même le nouveau simulateur, mais nous n'en avons pas eu le temps." />

      <Title title="Dataset filtrée: uniformiser la sortie" size="md" />
      <Paragraph text="Une caractéristique importante sur une bonne dataset est d'avoir une sortie uniforme. Par exemple dans le cas d'un label one-hot, il faut idéalement avoir autant de sorties attendues pour chaque classe. Ce n'est malheureusement pas notre cas. Notre IA a donc appris un biais involontaire: une stratégie qui n'est pas souvent la meilleure ne sera tout simplement jamais choisie." />

      <Title title="Model shape idéal trouvé algorithmiquement" size="md" />
      <Paragraph text="Nous n'avons pas de recherches scientifiques appuyant notre théroie, mais nous pensons que générer des 'model shape' en brute force jusqu'à tomber sur un modèle qui converge peut aider à trouver la forme idéale de notre modèle dense. Le modèle sur lequel nous avons fini par nous mettre d'accord est un FC relu avec 4 hidden layers de largeur 64." />
      <Paragraph text="Nous vous proposons ici de regarder un résultat qui prouve le concept, sur trois modèles différents en termes de profondeur et de largeur." />
      <Image src={SHAPECMP} alignSelf={"center"} />
      <Image src={DEEP_SPIKES} alignSelf={"center"} />
      <Paragraph text="La légende se lit comme ceci: (hidden_layers, hidden_layer_width, activation_function)" />
      <Paragraph text="Comme nous pouvons l'observer, en augmentant trop la profondeur on se retrouve parfois avec des pertes géantes (nous avons même dû supprimer quelques entrées à >18k). Cela ne signifie pas forcémeent que le modèle n'est pas viable, simplement qu'il faut bien choisir l'époque la plus efficace pendant l'entraînement." />
      <Paragraph text="La fonction d'activation leaky_relu semble aussi être plus efficace ici, sûrment à cause du fait que le gradient s'annule trop vite avec relu." />
    </Container>
  );
};
