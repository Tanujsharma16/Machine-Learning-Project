import React, { useState } from "react";

function App() {
  const [formData, setFormData] = useState({
    Pregnancies: "",
    Glucose: "",
    BloodPressure: "",
    SkinThickness: "",
    Insulin: "",
    BMI: "",
    DiabetesPedigreeFunction: "",
    Age: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict?model=lr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...formData,
          Pregnancies: Number(formData.Pregnancies),
          Glucose: Number(formData.Glucose),
          BloodPressure: Number(formData.BloodPressure),
          SkinThickness: Number(formData.SkinThickness),
          Insulin: Number(formData.Insulin),
          BMI: Number(formData.BMI),
          DiabetesPedigreeFunction: Number(formData.DiabetesPedigreeFunction),
          Age: Number(formData.Age),
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP Error! Status: ${res.status}`);
      }

      const data = await res.json();
      console.log("✅ API Response:", data);
      setResult(data);
    } catch (error) {
      console.error("❌ Error:", error);
      setResult({ prediction: "❌ Backend error" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: "sans-serif", textAlign: "center", padding: "30px" }}>
      <h1>🩺 Diabetes Prediction App</h1>
      <form onSubmit={handleSubmit} style={{ display: "grid", gap: "10px", width: "300px", margin: "auto" }}>
        {Object.keys(formData).map((key) => (
          <input
            key={key}
            type="number"
            step="any"
            name={key}
            placeholder={key}
            value={formData[key]}
            onChange={handleChange}
            required
          />
        ))}
        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {result && (
        <div style={{ marginTop: "20px", background: "#f8f9fa", padding: "15px", borderRadius: "8px" }}>
          <h2>Prediction Result</h2>
          <p><strong>Result:</strong> {result.prediction}</p>
          <p><strong>Probability:</strong> {result.probability}</p>
          <p><strong>Model Used:</strong> {result.model_used}</p>
        </div>
      )}
    </div>
  );
}

export default App;
