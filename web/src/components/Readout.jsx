/**
 * One measured number with its label.
 *
 * `win` marks the number that carries the stage's claim. It is the
 * only one tinted, so tinting more than one per stage would make the
 * mark mean nothing.
 */
export default function Readout({ value, label, win = false }) {
  return (
    <div className={win ? "readout win" : "readout"}>
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}
