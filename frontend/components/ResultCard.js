export default function ResultCard({ soh, rul, cycles }) {
  return (
    <div className="bg-green-50 p-5 rounded-xl shadow-md text-center">
      <h2 className="text-xl font-bold mb-3">Prediction Results</h2>
      <p className="text-lg">🔋 <b>SOH:</b> {soh.toFixed(2)}%</p>
      <p className="text-lg">⏳ <b>RUL (Years):</b> {rul.toFixed(2)}</p>
      <p className="text-lg">🔁 <b>RUL Cycles:</b> {cycles.toFixed(2)}</p>
    </div>
  );
}
