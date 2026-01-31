pub mod audio;
pub mod wake;

use crate::voice::audio::AudioRecorder;
use crate::voice::wake::WakeWordDetector;
use pi_core::agent_types::AgentCommand;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::info;

pub struct VoiceManager {
    recorder: Arc<tokio::sync::Mutex<AudioRecorder>>,
    detector: Option<Arc<WakeWordDetector>>,
    agent_cmd_tx: mpsc::Sender<AgentCommand>,
    /// Cancellation token for the background wake-word listener
    listener_cancel: Option<CancellationToken>,
    /// Whether the wake-word listener is running
    is_wake_listening: bool,
    /// Whether a push-to-talk recording is in progress
    is_ptt_recording: bool,
    /// Directory for temporary WAV files
    media_dir: PathBuf,
}

impl VoiceManager {
    pub fn new(agent_cmd_tx: mpsc::Sender<AgentCommand>) -> Self {
        let media_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".pi-assistant")
            .join("media")
            .join("voice");

        Self {
            recorder: Arc::new(tokio::sync::Mutex::new(AudioRecorder::new())),
            detector: None,
            agent_cmd_tx,
            listener_cancel: None,
            is_wake_listening: false,
            is_ptt_recording: false,
            media_dir,
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

    /// Start the background wake-word listener.
    pub async fn start_listening(&mut self) -> anyhow::Result<()> {
        if self.is_wake_listening {
            return Ok(());
        }

        let mut recorder = self.recorder.lock().await;
        recorder.start()?;
        drop(recorder);

        let recorder_clone = self.recorder.clone();
        let detector_clone = self
            .detector
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Detector not initialized"))?;
        let agent_cmd_tx = self.agent_cmd_tx.clone();

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        tokio::spawn(async move {
            info!("Voice background listener started");
            loop {
                tokio::select! {
                    _ = cancel_clone.cancelled() => {
                        info!("Voice background listener cancelled");
                        break;
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(500)) => {
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
                }
            }
            // Stop the audio stream when listener exits
            let mut recorder = recorder_clone.lock().await;
            recorder.stop();
            info!("Voice background listener stopped and audio stream released");
        });

        self.listener_cancel = Some(cancel);
        self.is_wake_listening = true;
        Ok(())
    }

    /// Stop the background wake-word listener.
    pub async fn stop_listening(&mut self) {
        if let Some(cancel) = self.listener_cancel.take() {
            cancel.cancel();
        }
        self.is_wake_listening = false;
        info!("Voice listener stop requested");
    }

    /// Start a push-to-talk recording session.
    /// Audio is captured into the buffer until `push_to_talk_stop` is called.
    pub async fn push_to_talk_start(&mut self) -> anyhow::Result<()> {
        if self.is_ptt_recording {
            return Ok(());
        }

        // Stop wake-word listener if active (they share the recorder)
        if self.is_wake_listening {
            self.stop_listening().await;
            // Give the listener task time to release the recorder
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        let mut recorder = self.recorder.lock().await;
        // Clear any stale audio
        recorder.take_buffer();
        recorder.start()?;

        self.is_ptt_recording = true;
        info!("Push-to-talk recording started");
        Ok(())
    }

    /// Stop push-to-talk recording, save the audio as a WAV file, and return the path.
    pub async fn push_to_talk_stop(&mut self) -> anyhow::Result<PathBuf> {
        if !self.is_ptt_recording {
            return Err(anyhow::anyhow!("No push-to-talk recording in progress"));
        }

        let mut recorder = self.recorder.lock().await;
        let audio_data = recorder.take_buffer();
        recorder.stop();
        drop(recorder);

        self.is_ptt_recording = false;

        if audio_data.is_empty() {
            return Err(anyhow::anyhow!("No audio data captured"));
        }

        // Ensure media directory exists
        tokio::fs::create_dir_all(&self.media_dir).await?;

        // Generate WAV file path
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let wav_path = self.media_dir.join(format!("ptt_{}.wav", timestamp));

        // Write WAV file (16kHz mono f32 → 16-bit PCM)
        let wav_path_clone = wav_path.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };

            let mut writer = hound::WavWriter::create(&wav_path_clone, spec)?;
            for &sample in &audio_data {
                let clamped = sample.clamp(-1.0, 1.0);
                writer.write_sample((clamped * i16::MAX as f32) as i16)?;
            }
            writer.finalize()?;
            Ok::<PathBuf, anyhow::Error>(wav_path_clone)
        });

        let path = handle.await??;
        info!("Push-to-talk audio saved to {:?}", path);
        Ok(path)
    }

    /// Cancel current push-to-talk recording without transcribing.
    pub async fn push_to_talk_cancel(&mut self) -> anyhow::Result<()> {
        if !self.is_ptt_recording {
            return Ok(());
        }

        let mut recorder = self.recorder.lock().await;
        recorder.take_buffer();
        recorder.stop();
        drop(recorder);

        self.is_ptt_recording = false;
        info!("Push-to-talk recording cancelled and buffer cleared");
        Ok(())
    }

    /// Check if the wake-word listener is active.
    pub fn is_listening(&self) -> bool {
        self.is_wake_listening
    }

    /// Check if push-to-talk recording is active.
    pub fn is_ptt_recording(&self) -> bool {
        self.is_ptt_recording
    }
}
