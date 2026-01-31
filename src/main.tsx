import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles/index.css";

// Global error handler for startup crashes
window.onerror = function (message, source, lineno, colno, error) {
  console.error("Global error:", message, error);
  document.body.innerHTML = `<div style="color: red; padding: 20px; font-family: monospace;">
        <h1>Startup Error</h1>
        <pre>${message}\n${source}:${lineno}:${colno}\n${error?.stack || ""}</pre>
    </div>`;
};

try {
  const root = document.getElementById("root");
  if (!root) throw new Error("Root element not found");

  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
} catch (e) {
  console.error("Mount error:", e);
  document.body.innerHTML = `<div style="color: red; padding: 20px;"><h1>Mount Error</h1><pre>${String(e)}</pre></div>`;
}
