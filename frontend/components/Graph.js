export default function Graph({ title, src }) {
  return (
    <div className="bg-white p-5 rounded-xl shadow-md">
      <h2 className="text-xl font-bold mb-3">{title}</h2>
      <img src={src} alt={title} className="rounded-lg w-full" />
    </div>
  );
}
