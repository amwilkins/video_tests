use opencv::{
    prelude::*,
    videoio::*,
};
use std::sync::mpsc::Sender;
use std::thread;
use std::time::{Duration, Instant};

// create camera thread
pub fn spawn_camera(tx: Sender<opencv::core::Mat>) -> std::thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut cam = VideoCapture::new(0, CAP_ANY).expect("Camera not found");
        // let _ = cam.set(CAP_PROP_FRAME_WIDTH, 800.0); // Set frame width
        // let _ = cam.set(CAP_PROP_FRAME_HEIGHT, 448.0); // Set frame height
        // let _ = cam.set(CAP_PROP_FPS, 30.0);
        // limit to camera fps property
        let target_fps = cam.get(CAP_PROP_FPS).unwrap_or(30.0);
        let frame_period = Duration::from_secs_f64(1.0 / target_fps.max(30.0));

        let time = Instant::now();
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


