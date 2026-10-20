import { useEffect, useState } from "react";
import { fetchWalkthrough, listPapers } from "./api";

/**
 * Load one paper's walkthrough, or the first paper there is.
 *
 * Returns a single object rather than four loose values so a component
 * cannot render half a state — `data` is never set while `error` is,
 * and neither is set while `loading` is true.
 *
 * The paper id is read from ?paper= so a presenter can open the exact
 * paper they rehearsed on instead of whichever one happens to be first.
 */
export function useWalkthrough(paperId) {
  const [state, setState] = useState({ loading: true, error: null, data: null });

  useEffect(() => {
    // A paper can take a moment to load and the id can change while it
    // does. Without this the first response can land after the second
    // and overwrite it, which shows the wrong paper with no error.
    let current = true;

    async function load() {
      setState({ loading: true, error: null, data: null });
      try {
        const id = paperId ?? (await listPapers())[0];
        if (!id) {
          throw new Error(
            "No papers have been ingested yet. Drop a PDF on the front page first."
          );
        }
        const data = await fetchWalkthrough(id);
        if (current) setState({ loading: false, error: null, data: { ...data, id } });
      } catch (error) {
        if (current) setState({ loading: false, error: error.message, data: null });
      }
    }

    load();
    return () => {
      current = false;
    };
  }, [paperId]);

  return state;
}
