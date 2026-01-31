use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use tracing::{error, info};

pub struct AudioRecorder {
    stream: Option<SendSafeStream>,
    buffer: Arc<Mutex<Vec<f32>>>,
}

struct SendSafeStream(cpal::Stream);
unsafe impl Send for SendSafeStream {}
unsafe impl Sync for SendSafeStream {}

impl std::ops::Deref for SendSafeStream {
    type Target = cpal::Stream;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AudioRecorder {
    pub fn new() -> Self {
        Self {
            stream: None,
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn start(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let devices = host.input_devices()?;

        let device = devices
            .filter(|d| {
                d.name()
                    .map(|n| {
                        n.to_lowercase().contains("pulse") || n.to_lowercase().contains("pipewire")
                    })
                    .unwrap_or(false)
            })
            .next()
            .or_else(|| host.default_input_device())
            .ok_or_else(|| anyhow!("No suitable input device found"))?;

        info!("Using input device: {}", device.name()?);

        let config = device.default_input_config()?;
        let sample_format = config.sample_format();
        let config: cpal::StreamConfig = config.into();

        let buffer = self.buffer.clone();

        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut b) = buffer.lock() {
                        b.extend_from_slice(data);
                    }
                },
                |err| error!("Error in audio stream: {}", err),
                None,
            )?,
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if let Ok(mut b) = buffer.lock() {
                        b.extend(data.iter().map(|&s| s as f32 / i16::MAX as f32));
                    }
                },
                |err| error!("Error in audio stream: {}", err),
                None,
            )?,
            _ => return Err(anyhow!("Unsupported sample format")),
        };

        stream.play()?;
        self.stream = Some(SendSafeStream(stream));

        Ok(())
    }

    pub fn stop(&mut self) {
        self.stream = None;
    }

    pub fn take_buffer(&self) -> Vec<f32> {
        if let Ok(mut b) = self.buffer.lock() {
            let data = b.clone();
            b.clear();
            data
        } else {
            Vec::new()
        }
    }
}
