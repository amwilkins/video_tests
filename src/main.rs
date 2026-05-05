use opencv::{
    //core::{add_weighted, Mat, Point, Scalar, CV_8UC3},
    highgui,
    imgproc,
};
use std::sync::mpsc;
use std::time::{Duration, Instant};

mod camera;
mod color_detect;
mod key_commands;
mod overlay;
//mod mouse_callback;

mod prelude {
    pub use crate::camera::*;
    pub use crate::color_detect::*;
    pub use crate::key_commands::*;
    pub use crate::overlay::*;
    pub use crate::*;
    //pub use crate::mouse_callback::create_mouse_callback;
    pub use opencv::core::*;
    pub use opencv::prelude::*;
    pub use std::sync::*;
}

use prelude::*;

pub struct State {
    camera_enabled: bool,
    reset_color_mode: bool,
    color_overlay: ColorOverlay,
    camera_rows: i32,
    camera_cols: i32,
    camera_rx: mpsc::Receiver<Mat>,
}

impl State {
    fn new() -> Self {
        Self {
            camera_enabled: false,
            reset_color_mode: false,
            color_overlay: ColorOverlay::new(),
            camera_rows: 0,
            camera_cols: 0,
            camera_rx: mpsc::channel().1,
        }
    }
}

fn main() -> opencv::Result<()> {
    let mut state = State::new();

    let (tx, rx) = mpsc::channel();
    let _cam_handle = spawn_camera(tx);
    highgui::named_window("Screen", highgui::WINDOW_AUTOSIZE)?;

    // ~~~ SETUP ~~~ //
    let mut frame_count = 0u32;
    let mut last_time = Instant::now();
    let mut current_fps = 0.0;

    // // mouse callback
    // let mouse_cb = create_mouse_callback(color_mutex.clone());
    // highgui::set_mouse_callback("Screen", mouse_cb)?;

    // select detection colors
    let detect_colors = vec![&GREEN_RANGE, &BLUE_RANGE, &RED1_RANGE, &RED2_RANGE];

    //creating frames
    //let mut color_frame = Mat::default();
    let mut camera_frame = rx.recv().unwrap(); // blocking wait for first frame
    state.camera_rows = camera_frame.rows();
    state.camera_cols = camera_frame.cols();

    let mut output_frame = Mat::zeros(state.camera_rows, state.camera_cols, CV_8UC3)?.to_mat()?;
    let mut overlay_frame = Mat::zeros(state.camera_rows, state.camera_cols, CV_8UC3)?.to_mat()?;
    let mut color_frame = Mat::zeros(state.camera_rows, state.camera_cols, CV_8UC3)?.to_mat()?;

    // ~~~ draw loop ~~~ //
    loop {
        // receive camera frame
        if let Some(cam_rec) = rx.try_recv().ok() {
            camera_frame = cam_rec;
            output_frame = camera_frame.clone();
        
            detect_and_draw_color(&mut color_frame, &camera_frame, &detect_colors)?;
            // fps
            frame_count += 1;
            if last_time.elapsed() >= Duration::from_secs(1) {
                current_fps = frame_count as f64 / last_time.elapsed().as_secs_f64();
                frame_count = 0;
                last_time = Instant::now();
            }
        }

        overlay_frame = create_overlay(&state);

        // text to screen
        let text = format!(
            "Resolution: {}x{}\nFPS: {:.1}",
            state.camera_rows, state.camera_cols, current_fps,
        );

        // match overlay frame to output
        overlay_frame.set_to(&Scalar::all(0.0), &Mat::default())?;
        imgproc::put_text(
            &mut overlay_frame,
            &text,
            Point::new(16, 16),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::all(255.0),
            1,
            imgproc::LINE_AA,
            false,
        )
        .ok();

        // add color_overlay to output
        add_weighted(
            &color_frame,
            1.0,
            &overlay_frame,
            1.0,
            0.0,
            &mut output_frame,
            -1,
        )?;

        // combining frames to output_frame
        if state.camera_enabled {
            add_weighted(
                &output_frame.clone(),
                1.0,
                &camera_frame,
                1.0,
                0.0,
                &mut output_frame,
                -1,
            )?;
        }

        // show the combined image
        highgui::imshow("Screen", &output_frame)?;

        let key = highgui::wait_key(1)?;
        handle_keypress(key, &mut state);
    }
}

