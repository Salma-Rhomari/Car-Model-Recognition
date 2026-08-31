"use client";

import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatClassName = (name) =>
    name
      .split("_")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");

  const buttonText = loading ? "Analyzing" : "Predict";

  return (
    <main className="min-h-screen bg-black text-white flex items-center justify-center px-6 py-16 relative overflow-hidden">
      {/* Ambient glow background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-emerald-500/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 right-0 w-[400px] h-[400px] bg-emerald-400/10 rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10 w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-10">
          <p className="text-xs tracking-[0.3em] text-neutral-500 uppercase mb-3">
            AI Recognition
          </p>
          <h1 className="text-3xl font-semibold tracking-tight">
            Car Model Recognition
          </h1>
          <p className="text-neutral-400 mt-2 text-sm">
            A deep learning model trained to identify car make, model, and year — with visual insight into how it makes each prediction.
          </p>
        </div>

        {/* Card */}
        <div className="rounded-2xl border border-neutral-800 bg-gradient-to-b from-neutral-900/80 to-black/80 backdrop-blur-xl p-8 shadow-2xl shadow-black/50">
          {/* Upload zone */}
          <label
            htmlFor="file-upload"
            className="group relative flex flex-col items-center justify-center w-full h-56 rounded-xl border border-dashed border-neutral-700 hover:border-emerald-500/50 bg-neutral-950/50 cursor-pointer transition-colors overflow-hidden"
          >
            {preview ? (
              <img
                src={preview}
                alt="Preview"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="flex flex-col items-center gap-2 px-4 text-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-8 h-8 text-neutral-500 group-hover:text-emerald-400 transition-colors"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 16.5V9.75m0 0l-3.75 3.75M12 9.75l3.75 3.75M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3.75 3.75 0 014.977 4.34A4.501 4.501 0 0118 19.5H6.75z"
                  />
                </svg>
                <p className="text-sm text-neutral-400">
                  <span className="text-white font-medium">
                    Click to upload
                  </span>{" "}
                  a car photo
                </p>
                <p className="text-xs text-neutral-600">JPG or PNG</p>
              </div>
            )}
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          {/* Predict button */}
          <div className="btn-wrapper mt-6">
            <button onClick={handleSubmit} disabled={!file || loading} className="btn">
              <svg
                className="btn-svg"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z"
                />
              </svg>
              <div className="txt-wrapper">
                {buttonText.split("").map((char, i) => (
                  <span
                    key={i}
                    className="btn-letter"
                    style={{ animationDelay: `${i * 0.08}s` }}
                  >
                    {char}
                  </span>
                ))}
              </div>
            </button>
          </div>

          {/* Error */}
          {error && (
            <p className="mt-4 text-sm text-red-400 text-center">{error}</p>
          )}

          {/* Result */}
          {result && (
            <div className="mt-6 rounded-xl border border-neutral-800 bg-neutral-950/70 p-5">
              <p className="text-xs tracking-widest text-neutral-500 uppercase mb-1">
                Predicted
              </p>
              <p className="text-xl font-semibold mb-4">
                {formatClassName(result.predicted_class)}
              </p>

              <div className="flex items-center justify-between text-sm text-neutral-400 mb-2">
                <span>Confidence</span>
                <span className="text-white font-medium">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-all duration-500"
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>

              {result.gradcam_image && (
                <div className="mt-5 pt-5 border-t border-neutral-800">
                  <p className="text-xs tracking-widest text-neutral-500 uppercase mb-3">
                    Where the model looked
                  </p>
                  <img
                    src={result.gradcam_image}
                    alt="Grad-CAM visualization"
                    className="w-full rounded-lg border border-neutral-800"
                  />
                </div>
              )}
            </div>
          )}
        </div>

        <p className="text-center text-xs text-neutral-600 mt-8">
          Powered by ResNet50 · Trained on the Stanford Cars Dataset
        </p>
      </div>
    </main>
  );
}