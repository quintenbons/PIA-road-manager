import { TabList, Tabs, Tab, TabPanels, TabPanel } from "@chakra-ui/tabs";
import { theme } from "@chakra-ui/theme";

export type MultipleTabsProps = {
  childrens: {
    [key: string]: React.ReactNode;
  };
};

export const MultipleTabs = (props: MultipleTabsProps) => {
  return (
    <Tabs variant="enclosed">
      <TabList>
        {Object.keys(props.childrens).map((key) => {
          return (
            <Tab
            sx={{
                _selected: {
                  color: theme.colors.green[400],
                  borderColor: theme.colors.green[400],
                  borderBottomWidth: 0,
                },
              }}
            >
              {key}
            </Tab>
          );
        })}
      </TabList>
      <TabPanels>
        {Object.values(props.childrens).map((value) => {
          return <TabPanel>{value}</TabPanel>;
        })}
      </TabPanels>
    </Tabs>
  );
};
