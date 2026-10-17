import { createContext, useCallback, useContext, useEffect, useState } from "react";

const LightboxContext = createContext(() => {});

/** Call this from any stage to open an image over the page. */
export const useZoom = () => useContext(LightboxContext);

/**
 * A page image opened over the walkthrough rather than in a tab of its
 * own. A raw bitmap in a new tab has a dead back button, because
 * nothing preceded it; here Escape or a click returns the reader to
 * the stage they were reading.
 */
export function LightboxProvider({ children }) {
  const [shown, setShown] = useState(null);
  const open = useCallback((src, alt) => setShown({ src, alt }), []);
  const close = useCallback(() => setShown(null), []);

  useEffect(() => {
    if (!shown) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    // The page behind must not scroll while the image is over it.
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [shown, close]);

  return (
    <LightboxContext.Provider value={open}>
      {children}
      {shown && (
        <div
          className="zoom"
          role="dialog"
          aria-modal="true"
          aria-label={shown.alt}
          onClick={close}
        >
          <button type="button" className="shut" onClick={close}>
            Close
          </button>
          <img src={shown.src} alt={shown.alt} />
        </div>
      )}
    </LightboxContext.Provider>
  );
}
