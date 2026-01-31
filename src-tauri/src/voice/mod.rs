pub mod audio;
pub mod wake;

use crate::voice::audio::AudioRecorder;
use crate::voice::wake::WakeWordDetector;
use pi_core::agent_types::AgentCommand;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::info;

pub struct VoiceManager {
    recorder: Arc<tokio::sync::Mutex<AudioRecorder>>,
    detector: Option<Arc<WakeWordDetector>>,
    agent_cmd_tx: mpsc::Sender<AgentCommand>,
}

impl VoiceManager {
    pub fn new(agent_cmd_tx: mpsc::Sender<AgentCommand>) -> Self {
        Self {
            recorder: Arc::new(tokio::sync::Mutex::new(AudioRecorder::new())),
            detector: None,
            agent_cmd_tx,
        }
    }

    pub async fn init_detector(
        &mut self,
        model_path: std::path::PathBuf,
        sample_rate: f32,
    ) -> anyhow::Result<()> {
        let detector = WakeWordDetector::new(model_path, sample_rate)?;
        self.detector = Some(Arc::new(detector));
        Ok(())
    }

    pub async fn start_listening(&self) -> anyhow::Result<()> {
        let mut recorder = self.recorder.lock().await;
        recorder.start()?;

        let recorder_clone = self.recorder.clone();
        let detector_clone = self
            .detector
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Detector not initialized"))?;
        let agent_cmd_tx = self.agent_cmd_tx.clone();

        tokio::spawn(async move {
            info!("Voice background listener started");
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                let data = {
                    let recorder = recorder_clone.lock().await;
                    recorder.take_buffer()
                };

                if !data.is_empty() && detector_clone.listen(&data) {
                    info!("Wake word detected! Triggering agent...");
                    let _ = agent_cmd_tx
                        .send(AgentCommand::ChatMessage {
                            content: "Hey Pi".to_string(),
                            provider: None,
                            model_id: None,
                        })
                        .await;
                }
            }
        });

        Ok(())
    }
}
