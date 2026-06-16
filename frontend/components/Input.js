export default function Input({ label, value, onChange }) {
  return (
    <div className="mb-4">
      <label className="block mb-1 font-medium">{label}</label>
      <input
        type="number"
        value={value}
        onChange={onChange}
        className="w-full p-2 border rounded-lg"
      />
    </div>
  );
}
