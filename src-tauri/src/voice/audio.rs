use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use tracing::{error, info};

// File descriptor redirection for suppressing library-level noise
#[cfg(target_os = "linux")]
struct SilenceStderr {
    backup_fd: i32,
}

#[cfg(target_os = "linux")]
impl SilenceStderr {
    fn new() -> Self {
        unsafe {
            let stderr_fd = 2;
            let backup_fd = libc_dup(stderr_fd);

            // O_WRONLY = 1
            let null_path = "/dev/null\0";
            let null_fd = libc_open(null_path.as_ptr() as *const i8, 1);

            if null_fd >= 0 {
                libc_dup2(null_fd, stderr_fd);
                libc_close(null_fd);
            }

            Self { backup_fd }
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for SilenceStderr {
    fn drop(&mut self) {
        if self.backup_fd >= 0 {
            unsafe {
                libc_dup2(self.backup_fd, 2);
                libc_close(self.backup_fd);
            }
        }
    }
}

extern "C" {
    #[link_name = "dup"]
    fn libc_dup(fd: i32) -> i32;
    #[link_name = "dup2"]
    fn libc_dup2(oldfd: i32, newfd: i32) -> i32;
    #[link_name = "close"]
    fn libc_close(fd: i32) -> i32;
    #[link_name = "open"]
    fn libc_open(path: *const i8, flags: i32) -> i32;
}

// Handle ALSA error messages by silencing them (backup mechanism)
#[cfg(target_os = "linux")]
fn silence_alsa_errors() {
    use std::os::raw::{c_char, c_int};

    // We declare the handler without variadic arguments for compatibility with Rust function definition
    type AlsaErrorHandler =
        unsafe extern "C" fn(*const c_char, c_int, *const c_char, c_int, *const c_char);

    extern "C" {
        // We tell Rust the function takes our non-variadic handler
        fn snd_lib_error_set_handler(handler: Option<AlsaErrorHandler>) -> c_int;
    }

    unsafe extern "C" fn dummy_error_handler(
        _file: *const c_char,
        _line: c_int,
        _function: *const c_char,
        _err: c_int,
        _fmt: *const c_char,
    ) {
    }

    unsafe {
        let _ = snd_lib_error_set_handler(Some(dummy_error_handler));
    }
}

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

impl Default for AudioRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioRecorder {
    pub fn new() -> Self {
        #[cfg(target_os = "linux")]
        silence_alsa_errors();

        Self {
            stream: None,
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn start(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let host_id = host.id();
        info!("Using audio host: {:?}", host_id);

        let device = {
            #[cfg(target_os = "linux")]
            let _silence = SilenceStderr::new();

            let mut devices = host.input_devices()?;

            devices
                .find(|d| {
                    d.name()
                        .map(|n| {
                            let name = n.to_lowercase();
                            // Prefer pulse/pipewire devices, avoid OSS (/dev/dsp) which is noisy
                            (name.contains("pulse")
                                || name.contains("pipewire")
                                || name.contains("default"))
                                && !name.contains("oss")
                        })
                        .unwrap_or(false)
                })
                .or_else(|| host.default_input_device())
                .ok_or_else(|| anyhow!("No suitable input device found"))?
        };

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
