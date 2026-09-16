//! Tegra hardware H.264 through the vendor V4L2 encoder.
//!
//! Tegra ships no `libnvidia-encode`, so the NVENC backend cannot open it, it publishes no render
//! node driver for the VA-API one to probe, and its encoder is not a plain V4L2 M2M node either:
//! the node is `/dev/nvhost-msenc` on JetPack 4 and `/dev/v4l2-nvenc` on JetPack 6, and that
//! JetPack 6 node is a `/dev/null` placeholder whose `open` the vendor `libnvv4l2.so` intercepts.
//! The library is therefore loaded at runtime, the way `nvenc.rs` loads `libcuda` and
//! `avcodec.rs` loads `libva`, and the encoder is driven with ordinary V4L2 ioctls. Nothing is
//! added to the build.
//!
//! One host frame becomes one access unit like this:
//!
//! 1. `Raw2NvBuffer` copies the packed frame into a pitch-linear surface. That copy is the floor
//!    on Tegra: there is no zero-copy X11 capture to import instead.
//! 2. `NvBufferTransform` converts the surface to block-linear NV12 on the VIC block, which costs
//!    the CPU nothing.
//! 3. The NV12 surface is queued on the output plane as a dmabuf and the access unit comes back on
//!    the capture plane, which is mapped through the fd `VIDIOC_EXPBUF` hands back: mapping the
//!    encoder fd itself answers ENODEV, and the vendor's own NvBuffer does the same thing.
//!
//! The encoder returns a unit one or two frames after the frame that produced it, so the frame
//! number travels in the buffer timestamp and the wire header is written from what comes back.
//!
//! The ioctl numbers and structure layouts were read off a target's headers rather than written
//! from memory, and `abi_matches` checks the sizes those numbers encode.

use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::mem::size_of;
use std::sync::OnceLock;
use std::ptr;

use libloading::{Library, Symbol};

use super::codec::{h264_frame_type, push_video_header, Codec, VIDEO_HEADER_LEN};
use super::reference::Reference;
use crate::RustCaptureSettings;

const V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE: u32 = 9;
const V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE: u32 = 10;
const V4L2_MEMORY_MMAP: u32 = 1;
const V4L2_MEMORY_DMABUF: u32 = 4;
const V4L2_PIX_FMT_NV12M: u32 = 0x3231_4d4e;
const V4L2_PIX_FMT_H264: u32 = 0x3436_3248;
/// The driver copies this buffer's timestamp to the access unit it produces.
const V4L2_BUF_FLAG_TIMESTAMP_COPY: u32 = 0x4000;

const VIDIOC_S_FMT: u64 = 0xc0d0_5605;
const VIDIOC_REQBUFS: u64 = 0xc014_5608;
const VIDIOC_QUERYBUF: u64 = 0xc058_5609;
const VIDIOC_QBUF: u64 = 0xc058_560f;
const VIDIOC_EXPBUF: u64 = 0xc040_5610;
const VIDIOC_DQBUF: u64 = 0xc058_5611;
const VIDIOC_STREAMON: u64 = 0x4004_5612;
const VIDIOC_STREAMOFF: u64 = 0x4004_5613;
const VIDIOC_S_EXT_CTRLS: u64 = 0xc020_5648;

const CID_BITRATE: u32 = 0x0099_09cf;
const CID_BITRATE_MODE: u32 = 0x0099_09ce;
const CID_H264_PROFILE: u32 = 0x0099_0a6b;
const CID_IDR_INTERVAL: u32 = 0x0099_0b02;
const CID_VBV_SIZE: u32 = 0x0099_0b13;
const CID_INSERT_SPS_PPS_AT_IDR: u32 = 0x0099_0b17;
const CID_HW_PRESET: u32 = 0x0099_0b1c;
const CID_INSERT_VUI: u32 = 0x0099_0b22;
const CID_MAX_PERFORMANCE: u32 = 0x0099_0b2a;
const CID_POC_TYPE: u32 = 0x0099_0b33;
const CID_FORCE_IDR_FRAME: u32 = 0x0099_0b37;

/// Every encoder control belongs to this class; the driver ignores a request that
/// arrives without it, silently, which is how a profile and a VUI went missing.
const V4L2_CTRL_CLASS_MPEG: u32 = 0x0099_0000;
const BITRATE_MODE_CBR: i32 = 1;
const H264_PROFILE_MAIN: i32 = 2;
const HW_PRESET_ULTRAFAST: i32 = 1;

const NVBUF_PAYLOAD_SURF_ARRAY: i32 = 0;
const NVBUF_LAYOUT_PITCH: i32 = 0;
const NVBUF_LAYOUT_BLOCK_LINEAR: i32 = 1;
const NVBUF_COLOR_ABGR32: i32 = 17;
const NVBUF_COLOR_XRGB32: i32 = 18;
const NVBUF_COLOR_NV12: i32 = 5;
const NVBUF_TAG_NONE: i32 = 0;
const NVBUF_TAG_VIDEO_ENC: i32 = 4608;
const NVBUF_TRANSFORM_FILTER: u32 = 4;
const NVBUF_FILTER_SMART: u32 = 4;

const OUTPUT_BUFFERS: usize = 4;
const CAPTURE_BUFFERS: usize = 4;
const ENCODER_NODES: [&str; 2] = ["/dev/v4l2-nvenc", "/dev/nvhost-msenc"];

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PlaneFormat {
    sizeimage: u32,
    bytesperline: u32,
    reserved: [u16; 6],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PixFormatMplane {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    colorspace: u32,
    plane_fmt: [PlaneFormat; 8],
    num_planes: u8,
    flags: u8,
    enc: u8,
    quantization: u8,
    xfer_func: u8,
    reserved: [u8; 7],
}

#[repr(C)]
struct Format {
    type_: u32,
    _pad: u32,
    pix_mp: PixFormatMplane,
    _tail: [u8; 8],
}

#[repr(C)]
#[derive(Default)]
struct RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Plane {
    bytesused: u32,
    length: u32,
    m: u64,
    data_offset: u32,
    reserved: [u32; 11],
}

#[repr(C)]
struct Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    _pad: u32,
    timestamp: [i64; 2],
    timecode: [u32; 4],
    sequence: u32,
    memory: u32,
    m: u64,
    length: u32,
    reserved2: u32,
    reserved: u32,
    _tail: u32,
}

#[repr(C)]
#[derive(Default)]
struct ExportBuffer {
    type_: u32,
    index: u32,
    plane: u32,
    flags: u32,
    fd: i32,
    reserved: [u32; 11],
}

#[repr(C, packed)]
struct ExtControl {
    id: u32,
    size: u32,
    reserved2: u32,
    value: i64,
}

#[repr(C)]
struct ExtControls {
    which: u32,
    count: u32,
    error_idx: u32,
    request_fd: i32,
    reserved: u32,
    _pad: u32,
    controls: *mut ExtControl,
}

#[repr(C)]
#[derive(Default)]
struct NvBufferCreateParams {
    width: i32,
    height: i32,
    payload_type: i32,
    memsize: i32,
    layout: i32,
    color_format: i32,
    nvbuf_tag: i32,
}

/// The surface format that matches the host frame's byte order. Checked on hardware by encoding
/// a frame of pure red and decoding it back: B,G,R,A bytes come out right as `XRGB32`, and the
/// mirrored R,G,B,A order as `ABGR32`, whatever the names suggest.
fn staging_format(rgba: bool) -> i32 {
    if rgba { NVBUF_COLOR_ABGR32 } else { NVBUF_COLOR_XRGB32 }
}

/// The sizes the ioctl numbers above encode, checked against what this build lays out.
fn abi_matches() -> Result<(), String> {
    let expected = [
        ("v4l2_format", size_of::<Format>(), 208),
        ("v4l2_requestbuffers", size_of::<RequestBuffers>(), 20),
        ("v4l2_buffer", size_of::<Buffer>(), 88),
        ("v4l2_plane", size_of::<Plane>(), 64),
        ("v4l2_exportbuffer", size_of::<ExportBuffer>(), 64),
        ("v4l2_ext_control", size_of::<ExtControl>(), 20),
        ("v4l2_ext_controls", size_of::<ExtControls>(), 32),
        ("NvBufferCreateParams", size_of::<NvBufferCreateParams>(), 28),
    ];
    for (name, got, want) in expected {
        if got != want {
            return Err(format!("{name} is {got} bytes here, the kernel ABI is {want}"));
        }
    }
    Ok(())
}

type V4l2Open = unsafe extern "C" fn(*const c_char, c_int) -> c_int;
type V4l2Ioctl = unsafe extern "C" fn(c_int, u64, *mut c_void) -> c_int;
type V4l2Close = unsafe extern "C" fn(c_int) -> c_int;
type NvCreate = unsafe extern "C" fn(*mut c_int, *const NvBufferCreateParams) -> c_int;
type NvRaw2Buf = unsafe extern "C" fn(*const u8, c_uint, c_int, c_int, c_int) -> c_int;
type NvTransform = unsafe extern "C" fn(c_int, c_int, *mut c_void) -> c_int;
type NvSync = unsafe extern "C" fn(c_int, c_uint, *mut *mut c_void) -> c_int;
type NvDestroy = unsafe extern "C" fn(c_int) -> c_int;

pub struct Vendor {
    _v4l2: Library,
    _nvbuf: Library,
    open: V4l2Open,
    ioctl: V4l2Ioctl,
    close: V4l2Close,
    create: NvCreate,
    raw2buf: NvRaw2Buf,
    transform: NvTransform,
    sync: NvSync,
    destroy: NvDestroy,
}

impl Vendor {
    /// Load both vendor libraries, plain name first so the loader's own search applies, then the
    /// L4T directory for images that do not put it on the default path.
    unsafe fn load() -> Result<Self, String> {
        let open_lib = |name: &str| -> Result<Library, String> {
            Library::new(name)
                .or_else(|_| Library::new(format!("/usr/lib/aarch64-linux-gnu/tegra/{name}")))
                .map_err(|e| format!("{name} is not loadable: {e}"))
        };
        let v4l2 = open_lib("libnvv4l2.so")?;
        let nvbuf = open_lib("libnvbuf_utils.so")?;
        let sym = |lib: &Library, name: &[u8]| -> Result<*const c_void, String> {
            let s: Symbol<*const c_void> = lib
                .get(name)
                .map_err(|e| format!("{} is missing: {e}", String::from_utf8_lossy(name)))?;
            Ok(*s)
        };
        Ok(Self {
            open: std::mem::transmute(sym(&v4l2, b"v4l2_open\0")?),
            ioctl: std::mem::transmute(sym(&v4l2, b"v4l2_ioctl\0")?),
            close: std::mem::transmute(sym(&v4l2, b"v4l2_close\0")?),
            create: std::mem::transmute(sym(&nvbuf, b"NvBufferCreateEx\0")?),
            raw2buf: std::mem::transmute(sym(&nvbuf, b"Raw2NvBuffer\0")?),
            transform: std::mem::transmute(sym(&nvbuf, b"NvBufferTransform\0")?),
            sync: std::mem::transmute(sym(&nvbuf, b"NvBufferMemSyncForDevice\0")?),
            destroy: std::mem::transmute(sym(&nvbuf, b"NvBufferDestroy\0")?),
            _v4l2: v4l2,
            _nvbuf: nvbuf,
        })
    }
}

/// Whether this host has the Tegra encoder: both vendor libraries load and a node opens. Probed
/// once, because a failure here is a property of the machine and not of the session.
pub fn available() -> bool {
    static PROBED: OnceLock<bool> = OnceLock::new();
    *PROBED.get_or_init(|| {
        let Some(vendor) = vendor() else { return false };
        for node in ENCODER_NODES {
            let Ok(path) = CString::new(node) else { continue };
            let fd = unsafe { (vendor.open)(path.as_ptr(), libc::O_RDWR) };
            if fd >= 0 {
                unsafe { (vendor.close)(fd) };
                return true;
            }
        }
        false
    })
}

/// The vendor libraries, loaded once for the process. They do not survive being unloaded and
/// loaded again: a second `dlopen` after the first handle is dropped fails, and a session then
/// falls back to software with nothing but a line in the log to say why.
fn vendor() -> Option<&'static Vendor> {
    static VENDOR: OnceLock<Option<Vendor>> = OnceLock::new();
    VENDOR
        .get_or_init(|| match unsafe { Vendor::load() } {
            Ok(vendor) => Some(vendor),
            Err(e) => {
                eprintln!("[pixelflux] Tegra vendor libraries unavailable: {e}");
                None
            }
        })
        .as_ref()
}

pub struct TegraEncoder {
    vendor: &'static Vendor,
    fd: c_int,
    width: i32,
    height: i32,
    row_bytes: usize,
    staging_fd: c_int,
    nv12_fd: [c_int; OUTPUT_BUFFERS],
    capture: [(*mut c_void, usize); CAPTURE_BUFFERS],
    queued: usize,
    scratch: Vec<u8>,
    omit_headers: bool,
    bitrate_bps: u32,
}

impl TegraEncoder {
    pub fn new(settings: &RustCaptureSettings, rgba: bool) -> Result<Self, String> {
        abi_matches()?;
        let (width, height) = (settings.width, settings.height);
        let fps = settings.target_fps.max(1.0);
        let bitrate_bps = (settings.video_bitrate_kbps.max(1) as u32).saturating_mul(1000);
        if width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0 {
            return Err(format!("the encoder needs even dimensions, got {width}x{height}"));
        }
        let vendor = vendor().ok_or("the Tegra vendor libraries are unavailable")?;

        let mut fd = -1;
        let mut opened = "";
        for node in ENCODER_NODES {
            let path = CString::new(node).unwrap();
            let candidate = unsafe { (vendor.open)(path.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK) };
            if candidate >= 0 {
                fd = candidate;
                opened = node;
                break;
            }
        }
        if fd < 0 {
            return Err(format!("no encoder node opened, tried {ENCODER_NODES:?}"));
        }

        let mut me = Self {
            vendor,
            fd,
            width,
            height,
            row_bytes: width as usize * 4,
            staging_fd: -1,
            nv12_fd: [-1; OUTPUT_BUFFERS],
            capture: [(ptr::null_mut(), 0); CAPTURE_BUFFERS],
            queued: 0,
            scratch: Vec::new(),
            omit_headers: settings.omit_stripe_headers,
            bitrate_bps,
        };
        if let Err(e) = me.setup(settings, fps, bitrate_bps, rgba) {
            return Err(format!("{opened}: {e}"));
        }
        Ok(me)
    }

    fn ioctl<T>(&self, request: u64, arg: &mut T, what: &str) -> Result<(), String> {
        let rc = unsafe { (self.vendor.ioctl)(self.fd, request, arg as *mut T as *mut c_void) };
        if rc < 0 {
            return Err(format!("{what} failed: {}", std::io::Error::last_os_error()));
        }
        Ok(())
    }

    /// Set one encoder control. `VIRTUALBUFFER_SIZE` is compound: the driver reads the value
    /// through a pointer in the same union, and passing it inline makes it read whatever address
    /// the number happens to name, which silently wrecks the rest of the configuration.
    fn set_control(&self, id: u32, value: i64, what: &str) -> Result<(), String> {
        let mut vbv = value as u32;
        let value = if id == CID_VBV_SIZE { &mut vbv as *mut u32 as i64 } else { value };
        let mut control = ExtControl { id, size: 0, reserved2: 0, value };
        let mut controls = ExtControls {
            which: V4L2_CTRL_CLASS_MPEG,
            count: 1,
            error_idx: 0,
            request_fd: 0,
            reserved: 0,
            _pad: 0,
            controls: &mut control,
        };
        self.ioctl(VIDIOC_S_EXT_CTRLS, &mut controls, what)
    }

    fn format(&self, type_: u32, pixelformat: u32, planes: u8, sizeimage: u32) -> Format {
        let mut plane_fmt = [PlaneFormat::default(); 8];
        plane_fmt[0].sizeimage = sizeimage;
        Format {
            type_,
            _pad: 0,
            pix_mp: PixFormatMplane {
                width: self.width as u32,
                height: self.height as u32,
                pixelformat,
                field: 1,
                colorspace: 0,
                plane_fmt,
                num_planes: planes,
                flags: 0,
                enc: 0,
                quantization: 0,
                xfer_func: 0,
                reserved: [0; 7],
            },
            _tail: [0; 8],
        }
    }

    fn setup(&mut self, settings: &RustCaptureSettings, fps: f64, bitrate_bps: u32, rgba: bool) -> Result<(), String> {
        let pixels = self.width as u32 * self.height as u32;
        let mut capture_format =
            self.format(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, V4L2_PIX_FMT_H264, 1, pixels.max(2 << 20));
        self.ioctl(VIDIOC_S_FMT, &mut capture_format, "S_FMT capture")?;
        let mut output_format =
            self.format(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, V4L2_PIX_FMT_NV12M, 2, 0);
        self.ioctl(VIDIOC_S_FMT, &mut output_format, "S_FMT output")?;

        // No infinite GOP here either: a session that asks for none gets ten seconds, long enough
        // not to spend bitrate on key frames and short enough to bound recovery when one is lost.
        let seconds = if settings.keyframe_interval_s > 0.0 { settings.keyframe_interval_s } else { 10.0 };
        let keyframe = ((fps * seconds) as i64).clamp(1, 600);
        let vbv = (bitrate_bps as f64 / fps.max(1.0)) as i64;
        self.set_control(CID_BITRATE, bitrate_bps as i64, "bitrate")?;
        self.set_control(CID_BITRATE_MODE, BITRATE_MODE_CBR as i64, "rate control mode")?;
        self.set_control(CID_H264_PROFILE, H264_PROFILE_MAIN as i64, "profile")?;
        self.set_control(CID_HW_PRESET, HW_PRESET_ULTRAFAST as i64, "preset")?;
        self.set_control(CID_MAX_PERFORMANCE, 1, "max performance")?;
        self.set_control(CID_INSERT_SPS_PPS_AT_IDR, 1, "SPS/PPS at IDR")?;
        self.set_control(CID_INSERT_VUI, 1, "VUI")?;
        self.set_control(CID_POC_TYPE, 2, "picture order count type")?;
        self.set_control(CID_IDR_INTERVAL, keyframe, "IDR interval")?;
        self.set_control(CID_VBV_SIZE, vbv, "VBV size")?;

        let mut request = RequestBuffers {
            count: OUTPUT_BUFFERS as u32,
            type_: V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
            memory: V4L2_MEMORY_DMABUF,
            ..Default::default()
        };
        self.ioctl(VIDIOC_REQBUFS, &mut request, "REQBUFS output")?;

        let mut staging = NvBufferCreateParams {
            width: self.width,
            height: self.height,
            payload_type: NVBUF_PAYLOAD_SURF_ARRAY,
            memsize: 0,
            layout: NVBUF_LAYOUT_PITCH,
            color_format: staging_format(rgba),
            nvbuf_tag: NVBUF_TAG_NONE,
        };
        if unsafe { (self.vendor.create)(&mut self.staging_fd, &staging) } < 0 {
            return Err("NvBufferCreateEx for the staging surface failed".into());
        }
        staging.layout = NVBUF_LAYOUT_BLOCK_LINEAR;
        staging.color_format = NVBUF_COLOR_NV12;
        staging.nvbuf_tag = NVBUF_TAG_VIDEO_ENC;
        for slot in 0..OUTPUT_BUFFERS {
            if unsafe { (self.vendor.create)(&mut self.nv12_fd[slot], &staging) } < 0 {
                return Err("NvBufferCreateEx for an NV12 surface failed".into());
            }
        }

        let mut request = RequestBuffers {
            count: CAPTURE_BUFFERS as u32,
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
            memory: V4L2_MEMORY_MMAP,
            ..Default::default()
        };
        self.ioctl(VIDIOC_REQBUFS, &mut request, "REQBUFS capture")?;
        for index in 0..CAPTURE_BUFFERS {
            let mut planes = [Plane::default(); 1];
            let mut buffer = self.buffer(
                V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
                V4L2_MEMORY_MMAP,
                index as u32,
                planes.as_mut_ptr(),
                1,
            );
            self.ioctl(VIDIOC_QUERYBUF, &mut buffer, "QUERYBUF capture")?;
            let length = planes[0].length as usize;
            let offset = planes[0].m as i64;
            let mut export = ExportBuffer {
                type_: V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
                index: index as u32,
                plane: 0,
                ..Default::default()
            };
            self.ioctl(VIDIOC_EXPBUF, &mut export, "EXPBUF capture")?;
            // The vendor's own NvBuffer::map maps this fd, not the encoder's: mapping the encoder
            // fd answers ENODEV on this node.
            let data = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    length,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    export.fd,
                    offset,
                )
            };
            unsafe { libc::close(export.fd) };
            if data == libc::MAP_FAILED {
                return Err(format!("mmap of a capture plane failed: {}", std::io::Error::last_os_error()));
            }
            self.capture[index] = (data, length);
            self.ioctl(VIDIOC_QBUF, &mut buffer, "QBUF capture")?;
        }

        let mut type_ = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        self.ioctl(VIDIOC_STREAMON, &mut type_, "STREAMON output")?;
        let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        self.ioctl(VIDIOC_STREAMON, &mut type_, "STREAMON capture")?;
        Ok(())
    }

    fn buffer(&self, type_: u32, memory: u32, index: u32, planes: *mut Plane, count: u32) -> Buffer {
        Buffer {
            index,
            type_,
            bytesused: 0,
            flags: 0,
            field: 0,
            _pad: 0,
            timestamp: [0; 2],
            timecode: [0; 4],
            sequence: 0,
            memory,
            m: planes as u64,
            length: count,
            reserved2: 0,
            reserved: 0,
            _tail: 0,
        }
    }

    pub fn codec(&self) -> Codec {
        Codec::H264
    }

    /// The VIC converts into NV12 and the encoder takes nothing else.
    pub fn is_fullcolor(&self) -> bool {
        false
    }

    /// Apply a live bitrate change. The control is writable while streaming on this path, unlike
    /// the same change through the vendor GStreamer element, which the driver accepts and ignores.
    pub fn reconfigure_rate(&mut self, settings: &RustCaptureSettings) -> Result<(), String> {
        let wanted = (settings.video_bitrate_kbps.max(1) as u32).saturating_mul(1000);
        if wanted == self.bitrate_bps {
            return Ok(());
        }
        self.set_control(CID_BITRATE, wanted as i64, "bitrate")?;
        self.set_control(CID_VBV_SIZE, (wanted as f64 / settings.target_fps.max(1.0)) as i64, "VBV size")?;
        self.bitrate_bps = wanted;
        Ok(())
    }

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
        let needed = stride * (height - 1) + self.row_bytes;
        if stride < self.row_bytes || pixels.len() < needed {
            return Err("input buffer too small".into());
        }
        // Raw2NvBuffer takes tight rows, so a padded frame is packed once into scratch.
        let source = if stride == self.row_bytes {
            pixels
        } else {
            self.scratch.resize(self.row_bytes * height, 0);
            for row in 0..height {
                let from = row * stride;
                self.scratch[row * self.row_bytes..(row + 1) * self.row_bytes]
                    .copy_from_slice(&pixels[from..from + self.row_bytes]);
            }
            &self.scratch
        };
        if unsafe {
            (self.vendor.raw2buf)(source.as_ptr(), 0, self.width, self.height, self.staging_fd)
        } < 0
        {
            return Err("Raw2NvBuffer failed".into());
        }

        let slot = if self.queued < OUTPUT_BUFFERS {
            self.queued
        } else {
            self.reclaim_output()?
        };
        let mut params = [0u8; 56];
        params[0..4].copy_from_slice(&NVBUF_TRANSFORM_FILTER.to_ne_bytes());
        params[8..12].copy_from_slice(&NVBUF_FILTER_SMART.to_ne_bytes());
        if unsafe {
            (self.vendor.transform)(
                self.staging_fd,
                self.nv12_fd[slot],
                params.as_mut_ptr() as *mut c_void,
            )
        } < 0
        {
            return Err("NvBufferTransform failed".into());
        }

        if force_idr {
            self.set_control(CID_FORCE_IDR_FRAME, 1, "force IDR")?;
        }
        let mut planes = [Plane::default(); 2];
        planes[0].m = self.nv12_fd[slot] as u64;
        planes[1].m = self.nv12_fd[slot] as u64;
        planes[0].bytesused = (self.width * self.height) as u32;
        planes[1].bytesused = (self.width * self.height / 2) as u32;
        let mut buffer = self.buffer(
            V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
            V4L2_MEMORY_DMABUF,
            slot as u32,
            planes.as_mut_ptr(),
            2,
        );
        buffer.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
        buffer.timestamp = [frame_number as i64, 0];
        self.ioctl(VIDIOC_QBUF, &mut buffer, "QBUF output")?;
        self.queued = (self.queued + 1).min(OUTPUT_BUFFERS);

        self.collect()
    }

    /// Wait for one queued output buffer to come back, which frees its slot. The node is
    /// non-blocking, so this is the one place that waits, and only when all of them are in flight.
    fn reclaim_output(&mut self) -> Result<usize, String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
        loop {
            let mut planes = [Plane::default(); 2];
            let mut buffer = self.buffer(
                V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
                V4L2_MEMORY_DMABUF,
                0,
                planes.as_mut_ptr(),
                2,
            );
            let rc = unsafe {
                (self.vendor.ioctl)(self.fd, VIDIOC_DQBUF, &mut buffer as *mut Buffer as *mut c_void)
            };
            if rc >= 0 {
                return Ok(buffer.index as usize);
            }
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EAGAIN) {
                return Err(format!("DQBUF output failed: {err}"));
            }
            if std::time::Instant::now() > deadline {
                return Err("the encoder did not return an output buffer in 500 ms".into());
            }
            let mut poll = libc::pollfd { fd: self.fd, events: libc::POLLOUT, revents: 0 };
            unsafe { libc::poll(&mut poll, 1, 20) };
        }
    }

    /// Take the access units the encoder has ready.
    fn collect(&mut self) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        loop {
            let mut planes = [Plane::default(); 1];
            let mut buffer = self.buffer(
                V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
                V4L2_MEMORY_MMAP,
                0,
                planes.as_mut_ptr(),
                1,
            );
            let rc = unsafe {
                (self.vendor.ioctl)(self.fd, VIDIOC_DQBUF, &mut buffer as *mut Buffer as *mut c_void)
            };
            if rc < 0 {
                break;
            }
            let index = buffer.index as usize;
            let length = planes[0].bytesused as usize;
            if length > 0 {
                let (data, _) = self.capture[index];
                let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, length) };
                if self.omit_headers {
                    out.extend_from_slice(bytes);
                } else {
                    out.reserve(VIDEO_HEADER_LEN + length);
                    push_video_header(
                        &mut out,
                        Codec::H264,
                        h264_frame_type(bytes),
                        buffer.timestamp[0] as u16,
                        0,
                        self.width as u16,
                        self.height as u16,
                        Reference::Untracked,
                    );
                    out.extend_from_slice(bytes);
                }
            }
            self.ioctl(VIDIOC_QBUF, &mut buffer, "QBUF capture")?;
        }
        Ok(out)
    }
}

/// The session is moved onto the encode thread and used from one thread at a time. What makes it
/// not `Send` on its own are the capture-plane mappings it owns and frees itself, as with the
/// NVENC and VA-API sessions next door.
unsafe impl Send for TegraEncoder {}

impl Drop for TegraEncoder {
    fn drop(&mut self) {
        let mut type_ = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        let _ = self.ioctl(VIDIOC_STREAMOFF, &mut type_, "STREAMOFF output");
        let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        let _ = self.ioctl(VIDIOC_STREAMOFF, &mut type_, "STREAMOFF capture");
        for (data, length) in self.capture {
            if !data.is_null() {
                unsafe { libc::munmap(data, length) };
            }
        }
        unsafe {
            (self.vendor.close)(self.fd);
            for fd in self.nv12_fd {
                if fd >= 0 {
                    (self.vendor.destroy)(fd);
                }
            }
            if self.staging_fd >= 0 {
                (self.vendor.destroy)(self.staging_fd);
            }
        }
    }
}
