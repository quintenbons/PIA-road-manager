import { Code, Container, theme, Image, Box } from "@chakra-ui/react";
import { BreadcrumbLnk } from "../components/BreadcrumbLnk";
import { DocumentDescriptor } from "../components/DocumentDescriptor";
import { Title } from "../components/Title";
import { Paragraph } from "../components/Paragraph";
import { ResultTab } from "../components/ResultTab";
import { AccordionParagraph } from "../components/AccordionParagraph";
import { DATASET_1 } from "../assets";
import BarChart from "../components/BarChart";
import LineChart from "../components/LineChart";
import training_data from "../data/first_training_data.json";
import scores_data from "../data/scores_per_strategy.json";

export const Mesures = () => {
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
        sections={[
          {
            name: "Home",
            href: "/",
          },
          {
            name: "Mesures",
            href: "/mesures",
          },
        ]}
      />
      <DocumentDescriptor
        title="Mesures"
        date="13 Décembre, 2023"
        authors={[
          "Julien Bourseau",
          "Luca Bitaudeau",
          "Quinten Bons",
          "Clément Juventin",
        ]}
      />
      <Title title="Estimation Préliminaire des Performances" size="md" />
      <Paragraph text="Il existe trois points critiques pour la performance de notre projet:" />
      <AccordionParagraph
        children={{
          "🧪 Estimation du temps de génération des datasets": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="Pour générer 100,000 entrées (I, E) (on rappelle que le but est de 1,000,000 d'entrées):" />
              <Paragraph text="Il faut pouvoir exécuter 15 minutes de simulation dans un intervalle de temps réel de l'ordre de la seconde. Si on utilise une solution naive, il nous faudra ([nb_strategies] + 1) x [temps de simul 15 minutes] pour générer 1 entrée (I, E)." />
              <Paragraph text="Avec 10 stratégies, on a donc 11 secondes par entrée, soit 305 heures. Sur 8 coeurs cela revient à 38 heures de génération, ce qui est loin d'être négligeable." />
              <Paragraph text="En pratique:" />
              <Paragraph text="À notre grande surprise, nous arrivons à générer en python 15 minutes de simulation en 1.4 secondes, sans avoir fait d'optimisation. Nous pensons pouvoir descendre en dessous de la seconde." />
              <Paragraph text="Générer un dataset est encore très coûteux. Comme nous avons en réalité 15 stratégies, nous mettons environ 60 heures à générer 100000 entrées (I, E)" />
            </Box>
          ),
          "🎰 Temps d'entraînement": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="L'entraînement consiste à faire un forward + backward sur des batch générés par un dataloader. La complexité de cette tâche est de l'ordre de la taille du dataset utilisé. En pratique, l'entraînement d'un DNN à 64 neuronnes par couche sur un dataset de 100,000 entrées peut se fait en 5 minutes sans même utiliser de GPU (mesuré avec un exercice de pathfinding)." />
              <Paragraph text="En pratique:" />
              <Paragraph text="Pas de mauvaise surprise, nous arrivons à entraîner l'IA dans des ordres de grandeurs négligeables face au temps de génération des datasets." />
            </Box>
          ),
          "🏧 Coût de fonctionnement": (
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: theme.space[4],
              }}
            >
              <Paragraph text="L'IA en pratique serait utilisée une fois par croisement, toutes les 15 minutes. La performance n'est pas requise temporellement. Cependant, une meilleure performance permet de réduire la consommation énergétique, ce qui est un des buts principaux du projet. En effet, si l'énergie consommée n'était pas importante, il suffirait de lancer notre algorithme naif toutes les 15 minutes sur la simulation." />
              <Paragraph text="En pratique:" />
              <Paragraph text="Le coût temporel forward de l'IA est vastement négligeable face au coût temporel de la simulation (de l'ordre de 0.01%). Cela prouve que le coût énergétique d'un forward est lui aussi très faible." />
            </Box>
          ),
        }}
      />
      <Title title="Génération de datasets de l'IA" size="md" />
      <Code colorScheme="green">
        100%|█████████████████████████████████████████| 96/96 [33:58/00:00,
        21.23s/it] 100%|█████████████████████████████████████████| 96/96
        [34:07/00:00, 21.33s/it] 100%|█████████████████████████████████████████|
        96/96 [34:14/00:00, 21.40s/it]
        100%|█████████████████████████████████████████| 96/96 [34:30/00:00,
        21.57s/it]
        <br />
      </Code>
      <Paragraph text="Voici l'output de notre script de génération en parallèle de dataset. Nous avons créé ici 96 entrées par coeur en 34 minutes." />
      <Paragraph text="Comme nous le voyons grâce à tqdm, nous pouvons générer 1 entrée de dataset en 21 secondes, sur 4 coeurs en simultané." />

      <Title title="Scores" size="md" />

      <BarChart data={scores_data} />
      <LineChart data={training_data} />

      <Title title="Performances du moteur" size="md" />
      <Title title="Benchmark #1" size="sm" />
      <ResultTab
        data={[
          {
            date: "14 Décembre, 2023",
            time: "150.15505409240723",
            commitHash: "d9a51775c2bd61c1a0e4e87a2ba83d9de7b6c463",
            granularity: "0.5",
          },
        ]}
        caption={"Exécution de la simulation sur le benchmark #1"}
      />
      <Image src={DATASET_1} width={"60%"} alignSelf={"center"} />
      <Paragraph text="Le benchmark #1 est une simulation à 7 neuds avec un débit de voiture supérieur à la capacité du réseau routier." />

      <Title title="Performances Engine & Graphique" size="md" />
      <Paragraph text="Avec la partie graphique de l'engine, la simulation est capable de tourner environ 128 fois plus rapidement qu'à la vitesse où le temps réel s'écoule." />

      <Title title="Configuration de test" size="md" />

      <AccordionParagraph
        children={{
          "lsb_release -a": (
            <Code colorScheme="green">
              Distributor ID: Ubuntu <br />
              Description: Ubuntu 22.04.3 <br />
              LTS Release: 22.04 <br />
              Codename: jammy
              <br />
            </Code>
          ),
          lscpu: (
            <Code colorScheme="green">
              Architecture: x86_64
              <br />
              CPU op-mode(s): 32-bit, 64-bit
              <br />
              Address sizes: 48 bits physical, 48 bits virtual
              <br />
              Byte Order: Little Endian
              <br />
              CPU(s): 16
              <br />
              On-line CPU(s) list: 0-15
              <br />
              Vendor ID: AuthenticAMD
              <br />
              Model name: AMD Ryzen 7 5800H with Radeon Graphics
              <br />
              CPU family: 25
              <br />
              Model: 80
              <br />
              Thread(s) per core: 2<br />
              Core(s) per socket: 8<br />
              Socket(s): 1<br />
              Stepping: 0<br />
              Frequency boost: enabled
              <br />
              CPU max MHz: 4462,5000
              <br />
              CPU min MHz: 1200,0000
              <br />
              BogoMIPS: 6388.23
              <br />
              Flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca
              cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext
              fxsr_opt pd pe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc
              cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma
              cx16 sse4 _1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand
              lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse
              3dnowprefe tch osvw ibs skinit wdt tce topoext perfctr_core
              perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate
              ssbd mba ibrs ibp b stibp vmmcall fsgsbase bmi1 avx2 smep bmi2
              erms invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni
              xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total
              cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd cppc arat
              npt lbrv svm_lock nrip _save tsc_scale vmcb_clean flushbyasid
              decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif
              v_spec_ctrl umip pku ospke vaes vpclmulqdq rdpid overflow_recov
              succor smca fsrm
              <br />
              Virtualization features: <br />
              Virtualization: AMD-V
              <br />
              Caches (sum of all): <br />
              L1d: 256 KiB (8 instances)
              <br />
              L1i: 256 KiB (8 instances)
              <br />
              L2: 4 MiB (8 instances)
              <br />
              L3: 16 MiB (1 instance)
              <br />
              NUMA: <br />
              NUMA node(s): 1<br />
              NUMA node0 CPU(s): 0-15
              <br />
              Vulnerabilities: <br />
              Gather data sampling: Not affected
              <br />
              Itlb multihit: Not affected
              <br />
              L1tf: Not affected
              <br />
              Mds: Not affected
              <br />
              Meltdown: Not affected
              <br />
              Mmio stale data: Not affected
              <br />
              Retbleed: Not affected
              <br />
              Spec rstack overflow: Mitigation; safe RET, no microcode
              <br />
              Spec store bypass: Mitigation; Speculative Store Bypass disabled
              via prctl
              <br />
              Spectre v1: Mitigation; usercopy/swapgs barriers and __user
              pointer sanitization
              <br />
              Spectre v2: Mitigation; Retpolines, IBPB conditional, IBRS_FW,
              STIBP always-on, RSB filling, PBRSB-eIBRS Not affected
              <br />
              Srbds: Not affected
              <br />
              Tsx async abort: Not affected
              <br />
            </Code>
          ),
        }}
      />

      <Title title="Pistes d'amélioration" size="md" />
      <Paragraph text="Nous n'avons pas réalisé de refactoring ou d'optimisation de la codebase. Nous pensons que cela pourrait largement améliorer les performances de l'engine et ainsi augmenter la vitesse de génération des datasets." />
      <Paragraph text="Nous aimerions aussi ajouter de nouveaux benchmarks. Nous sommes satisfait pour ce POC et nous pourrons toujours revenir aux commits." />
    </Container>
  );
};
