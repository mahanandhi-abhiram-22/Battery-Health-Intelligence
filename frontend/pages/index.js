import { useState } from "react";
import axios from "axios";
import Input from "../components/Input";
import Card from "../components/Card";
import Graph from "../components/Graph";
import ResultCard from "../components/ResultCard";

export default function Home() {
  const [voltage, setVoltage] = useState(3.7);
  const [current, setCurrent] = useState(1.0);
  const [temperature, setTemperature] = useState(25);
  const [cycle, setCycle] = useState(50);
  const [capacity, setCapacity] = useState(1);

  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    const res = await axios.post("http://127.0.0.1:8000/api/predict", {
      voltage,
      current,
      temperature,
      cycle_count: cycle,
      capacity
    });
    setResult(res.data);
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6 text-center">EV Battery Predictor</h1>

      <Card title="Enter Battery Parameters">
        <Input label="Voltage" value={voltage} onChange={e => setVoltage(e.target.value)} />
        <Input label="Current" value={current} onChange={e => setCurrent(e.target.value)} />
        <Input label="Temperature" value={temperature} onChange={e => setTemperature(e.target.value)} />
        <Input label="Cycle Count" value={cycle} onChange={e => setCycle(e.target.value)} />
        <Input label="Capacity" value={capacity} onChange={e => setCapacity(e.target.value)} />

        <button 
          onClick={handlePredict}
          className="mt-4 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
        >
          Predict
        </button>
      </Card>

      {result && (
        <ResultCard
          soh={result.SOH}
          rul={result.RUL}
          cycles={result.RUL_Cycles}
        />
      )}

      <div className="mt-8 space-y-6">
        <Graph title="Training Loss" src="http://127.0.0.1:8000/api/results/plot/training" />
        <Graph title="Parity Grid" src="http://127.0.0.1:8000/api/results/plot/parity_grid" />
      </div>
    </div>
  );
}
