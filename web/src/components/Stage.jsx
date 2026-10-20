/**
 * One pipeline box: its number, what it is called, one line on why it
 * exists, and whatever that box produced for this paper.
 *
 * The id is what the rail scrolls to and what the scroll spy reads
 * back, so it has to match the key in STAGES exactly.
 */
export default function Stage({ id, number, title, lede, children }) {
  return (
    <section className="stage" id={id} aria-labelledby={`${id}-h`}>
      {/* The last section is not a ninth box — it is the reason to
          believe the other seven — so it carries no number and no
          badge rather than a badge with nothing in it. */}
      <header className={number ? undefined : "unnumbered"}>
        {number && <span className="n">{number}</span>}
        <h3 id={`${id}-h`}>{title}</h3>
        <p className="lede">{lede}</p>
      </header>
      {children}
    </section>
  );
}
