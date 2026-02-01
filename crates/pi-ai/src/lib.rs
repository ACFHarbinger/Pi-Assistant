use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[derive(Serialize, Deserialize)]
pub struct Embedding {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[wasm_bindgen]
pub struct Model {
    // In a real implementation, this would hold the loaded Weights
    // For now, we simulate a simple projection
    weights: Tensor,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Model, JsValue> {
        // Initialize a random tensor to simulate model weights
        // In reality, we would fetch .safetensors bytes here
        let weights = Tensor::randn(0f32, 1f32, (384, 10), &Device::Cpu)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Model { weights })
    }

    pub fn predict(&self, _input: &str) -> Result<JsValue, JsValue> {
        // Simulate embedding generation
        // 1. Tokenize (mocked)
        // 2. Forward pass (mocked)

        let dummy_output = Tensor::randn(0f32, 1f32, (1, 384), &Device::Cpu)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let data = dummy_output
            .to_vec1::<f32>()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let embedding = Embedding {
            data,
            shape: vec![1, 384],
        };

        Ok(serde_wasm_bindgen::to_value(&embedding)?)
    }
}
