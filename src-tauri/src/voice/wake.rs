use anyhow::{anyhow, Result};
use std::path::PathBuf;
use tracing::{info, warn};
use vosk::{Model, Recognizer};

pub struct WakeWordDetector {
    model: Model,
    sample_rate: f32,
}

impl WakeWordDetector {
    pub fn new(model_path: PathBuf, sample_rate: f32) -> Result<Self> {
        if !model_path.exists() {
            return Err(anyhow!("Vosk model not found at {:?}", model_path));
        }

        let model = Model::new(
            model_path
                .to_str()
                .ok_or_else(|| anyhow!("Invalid model path"))?,
        )
        .ok_or_else(|| anyhow!("Failed to load Vosk model"))?;

        Ok(Self { model, sample_rate })
    }

    pub fn listen(&self, audio_data: &[f32]) -> bool {
        if audio_data.is_empty() {
            return false;
        }

        // Convert f32 to i16 for Vosk
        let i16_data: Vec<i16> = audio_data
            .iter()
            .map(|&s| (s * i16::MAX as f32) as i16)
            .collect();

        let mut recognizer = match Recognizer::new(&self.model, self.sample_rate) {
            Some(r) => r,
            None => {
                warn!("Failed to create Vosk recognizer");
                return false;
            }
        };

        recognizer.set_max_alternatives(0);
        recognizer.set_words(true);

        recognizer.accept_waveform(&i16_data);
        let result = recognizer.final_result();

        // Simple heuristic for "Hey Pi" or "Pi"
        // In a real implementation, we would parse the JSON and check for high confidence
        let text = result.single().map(|r| r.text).unwrap_or("");
        info!("Voice activity detected: {}", text);

        text.to_lowercase().contains("hey pi") || text.to_lowercase().contains("ok pi")
    }
}
