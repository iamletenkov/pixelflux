//! Tegra hardware H.264 through the vendor GStreamer elements.
//!
//! Tegra keeps NVENC behind `/dev/nvhost-msenc` and a proprietary V4L2 shim. It ships no
//! `libnvidia-encode`, so the NVENC backend cannot open it, and upstream FFmpeg's
//! `h264_v4l2m2m` does not drive it either. The vendor elements are the only path: `nvvidconv`
//! converts color on the VIC block, which costs the CPU nothing, and `nvv4l2h264enc` encodes
//! from the NVMM surfaces it produces. A session is one in-process pipeline:
//!
//! ```text
//! appsrc(BGRx or RGBA, host memory) -> nvvidconv -> video/x-raw(memory:NVMM),NV12
//!   -> nvv4l2h264enc -> h264parse -> appsink(byte-stream, alignment=au)
//! ```
//!
//! The encoder returns an access unit one or two frames after the frame that produced it, so the
//! frame number rides in the buffer PTS and is read back from the sample: the wire header names
//! the frame the unit came from, not the frame the caller last pushed.

use std::sync::OnceLock;

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;

use super::codec::{h264_frame_type, push_video_header, Codec, VIDEO_HEADER_LEN};
use super::reference::Reference;
use crate::RustCaptureSettings;

fn gst_ready() -> bool {
    static READY: OnceLock<bool> = OnceLock::new();
    *READY.get_or_init(|| match gst::init() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("[pixelflux] GStreamer init failed: {e}");
            false
        }
    })
}

/// Whether this host has the Tegra encoder, which is what both vendor elements being registered
/// means. A desktop with GStreamer installed answers false, and the ladder goes on to NVENC.
pub fn available() -> bool {
    gst_ready()
        && gst::ElementFactory::find("nvv4l2h264enc").is_some()
        && gst::ElementFactory::find("nvvidconv").is_some()
}

fn bitrate_bps(settings: &RustCaptureSettings) -> u32 {
    (settings.video_bitrate_kbps.max(1) as u32).saturating_mul(1000)
}

/// One frame of bits, the VBV size that keeps CBR from sawtoothing on a screen share.
fn vbv_size(settings: &RustCaptureSettings) -> u32 {
    ((bitrate_bps(settings) as f64 / settings.target_fps.max(1.0)) as u32).max(1)
}

/// Frames between IDRs. The element has no infinite GOP, so a session that asks for none gets
/// ten seconds: long enough not to spend bitrate on key frames, short enough to bound recovery
/// when a key frame request is lost.
fn keyframe_frames(settings: &RustCaptureSettings) -> u32 {
    let seconds = if settings.keyframe_interval_s > 0.0 { settings.keyframe_interval_s } else { 10.0 };
    ((settings.target_fps.max(1.0) * seconds) as u32).clamp(1, 600)
}

pub struct TegraEncoder {
    pipeline: gst::Pipeline,
    src: gst_app::AppSrc,
    sink: gst_app::AppSink,
    width: i32,
    height: i32,
    row_bytes: usize,
    frame_wait: gst::ClockTime,
    omit_headers: bool,
    bitrate_bps: u32,
}

impl TegraEncoder {
    pub fn new(settings: &RustCaptureSettings, rgba: bool) -> Result<Self, String> {
        if !gst_ready() {
            return Err("GStreamer is not available".into());
        }
        let (width, height) = (settings.width, settings.height);
        if width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0 {
            return Err(format!("the Tegra encoder needs even dimensions, got {width}x{height}"));
        }
        let fps = settings.target_fps.max(1.0);
        let description = format!(
            "appsrc name=src is-live=true do-timestamp=false format=time block=false \
             caps=video/x-raw,format={format},width={width},height={height},framerate={fps_n}/1000 \
             ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 \
             ! nvv4l2h264enc name=enc maxperf-enable=true preset-level=1 control-rate=1 \
               bitrate={bitrate} vbv-size={vbv} profile=2 num-B-Frames=0 poc-type=2 \
               insert-sps-pps=true insert-vui=true iframeinterval={keyframe} idrinterval={keyframe} \
               EnableTwopassCBR=false \
             ! h264parse config-interval=-1 \
             ! video/x-h264,stream-format=byte-stream,alignment=au \
             ! appsink name=sink sync=false max-buffers=4 drop=false",
            format = if rgba { "RGBA" } else { "BGRx" },
            fps_n = (fps * 1000.0).round() as i32,
            bitrate = bitrate_bps(settings),
            vbv = vbv_size(settings),
            keyframe = keyframe_frames(settings),
        );
        let pipeline = gst::parse::launch(&description)
            .map_err(|e| format!("failed to build the Tegra pipeline: {e}"))?
            .downcast::<gst::Pipeline>()
            .map_err(|_| "the Tegra pipeline is not a pipeline".to_string())?;
        let src = pipeline
            .by_name("src")
            .and_then(|e| e.downcast::<gst_app::AppSrc>().ok())
            .ok_or("the Tegra pipeline has no appsrc")?;
        let sink = pipeline
            .by_name("sink")
            .and_then(|e| e.downcast::<gst_app::AppSink>().ok())
            .ok_or("the Tegra pipeline has no appsink")?;
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| format!("failed to start the Tegra pipeline: {e}"))?;
        Ok(Self {
            pipeline,
            src,
            sink,
            width,
            height,
            row_bytes: (width as usize) * 4,
            frame_wait: gst::ClockTime::from_nseconds((1_000_000_000.0 / fps) as u64),
            omit_headers: settings.omit_stripe_headers,
            bitrate_bps: bitrate_bps(settings),
        })
    }

    pub fn codec(&self) -> Codec {
        Codec::H264
    }

    /// The VIC converts into NV12 and the element encodes nothing else.
    pub fn is_fullcolor(&self) -> bool {
        false
    }

    /// A live bitrate write is accepted by the element and ignored by the driver on L4T 32.x, so
    /// the caller is told to rebuild the session rather than handed a rate that never took.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        let wanted = bitrate_bps(settings);
        if wanted == self.bitrate_bps {
            return Ok(());
        }
        Err(format!(
            "the Tegra encoder cannot change bitrate in place ({} to {} bps)",
            self.bitrate_bps, wanted
        ))
    }

    /// Encode one packed host frame. An empty vector means the encoder is still holding every
    /// frame it was given, which the caller sends nothing for.
    pub fn encode_host(
        &mut self,
        pixels: &[u8],
        stride: usize,
        _rgba: bool,
        frame_number: u64,
        _qp: u32,
        force_idr: bool,
    ) -> Result<Vec<u8>, String> {
        let height = self.height as usize;
        let needed = if height == 0 {
            0
        } else {
            stride.checked_mul(height - 1).ok_or("stride overflow")? + self.row_bytes
        };
        if stride < self.row_bytes || pixels.len() < needed {
            return Err("input buffer too small".into());
        }
        if force_idr {
            let event = gst_video::UpstreamForceKeyUnitEvent::builder().all_headers(true).build();
            self.pipeline.send_event(event);
        }

        let mut buffer = gst::Buffer::with_size(self.row_bytes * height)
            .map_err(|_| "failed to allocate a buffer".to_string())?;
        {
            let buffer = buffer.get_mut().ok_or("the buffer is shared")?;
            buffer.set_pts(gst::ClockTime::from_nseconds(frame_number));
            let mut map = buffer
                .map_writable()
                .map_err(|_| "failed to map the buffer".to_string())?;
            let destination = map.as_mut_slice();
            if stride == self.row_bytes {
                destination.copy_from_slice(&pixels[..self.row_bytes * height]);
            } else {
                for row in 0..height {
                    let from = row * stride;
                    destination[row * self.row_bytes..(row + 1) * self.row_bytes]
                        .copy_from_slice(&pixels[from..from + self.row_bytes]);
                }
            }
        }
        self.src
            .push_buffer(buffer)
            .map_err(|e| format!("failed to push a frame into the Tegra pipeline: {e}"))?;

        self.pull_encoded()
    }

    /// Take the access units the encoder has ready. The first wait is one frame, which is what a
    /// session has to spare before it falls behind; the rest are taken without waiting, so a
    /// backlog drains in the same call.
    fn pull_encoded(&mut self) -> Result<Vec<u8>, String> {
        let mut output = Vec::new();
        let mut wait = self.frame_wait;
        while let Some(sample) = self.sink.try_pull_sample(wait) {
            wait = gst::ClockTime::ZERO;
            let Some(buffer) = sample.buffer() else { continue };
            let frame_id = buffer.pts().map(|pts| pts.nseconds()).unwrap_or(0) as u16;
            let map = buffer
                .map_readable()
                .map_err(|_| "failed to map an encoded buffer".to_string())?;
            let bytes = map.as_slice();
            if bytes.is_empty() {
                continue;
            }
            if self.omit_headers {
                output.extend_from_slice(bytes);
                continue;
            }
            output.reserve(VIDEO_HEADER_LEN + bytes.len());
            push_video_header(
                &mut output,
                Codec::H264,
                h264_frame_type(bytes),
                frame_id,
                0,
                self.width as u16,
                self.height as u16,
                Reference::Untracked,
            );
            output.extend_from_slice(bytes);
        }
        Ok(output)
    }
}

impl Drop for TegraEncoder {
    fn drop(&mut self) {
        let _ = self.src.end_of_stream();
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}
