import { useState } from "react";

export const useNavigation = () => {
  const [path, setPath] = useState<string>("/");

  return {
    path,
    setPath,
  };
};
