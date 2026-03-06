use opencv::{
    prelude::*,
    videoio::{CAP_ANY, CAP_PROP_FPS, VideoCapture},
};
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Duration;

// create camera thread
pub fn spawn_camera(tx: Sender<opencv::core::Mat>) -> std::thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut cam = VideoCapture::new(0, CAP_ANY).expect("Camera not found");
        // limit to camera fps property
        let target_fps = cam.get(CAP_PROP_FPS).unwrap_or(30.0);
        let frame_period = Duration::from_secs_f64(1.0 / target_fps.max(30.0));

        // send frame loop
        loop {
            let mut frame = opencv::core::Mat::default();
            if cam.read(&mut frame).unwrap_or(false) {
                tx.send(frame).ok();
            }
            // throttle speed
            thread::sleep(frame_period);
        }
    })
}
