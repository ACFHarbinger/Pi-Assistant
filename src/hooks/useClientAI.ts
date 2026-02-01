import { useEffect, useState } from "react";
import init, { Model, init_panic_hook } from "../wasm/pi-ai";
import { useAgentStore } from "../stores/agentStore";

export function useClientAI() {
  const { clientAI, setClientAIModel } = useAgentStore();
  const [isInitializing, setIsInitializing] = useState(false);

  useEffect(() => {
    // Only load if not loaded and not currently initializing
    // Check if model is already set in store to avoid double init
    if (!clientAI.isLoaded && !clientAI.model && !isInitializing) {
      setIsInitializing(true);
      const loadWasm = async () => {
        try {
          console.log("Initializing Client AI Wasm...");
          await init();
          init_panic_hook();
          const model = new Model();
          setClientAIModel(model);
          console.log("Client AI Model loaded successfully");
        } catch (e) {
          console.error("Failed to load Client AI Wasm:", e);
        } finally {
          setIsInitializing(false);
        }
      };
      loadWasm();
    }
  }, [clientAI.isLoaded, clientAI.model, isInitializing, setClientAIModel]);

  const predict = (input: string) => {
    if (clientAI.model) {
      try {
        return clientAI.model.predict(input);
      } catch (e) {
        console.error("Prediction failed:", e);
        return null;
      }
    }
    return null;
  };

  return {
    isLoaded: clientAI.isLoaded,
    predict,
  };
}
