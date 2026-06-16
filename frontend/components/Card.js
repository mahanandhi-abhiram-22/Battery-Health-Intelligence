export default function Card({ title, children }) {
  return (
    <div className="bg-white p-5 rounded-xl shadow-md mb-6">
      <h2 className="text-xl font-bold mb-3">{title}</h2>
      {children}
    </div>
  );
}
