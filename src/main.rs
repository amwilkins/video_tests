use opencv::{
    core::{CV_8UC3, Mat, Point, Scalar, add_weighted},
    highgui, imgproc,
    prelude::*,
};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod camera;

mod color_detect;
use crate::color_detect::{BLUE_RANGE, GREEN_RANGE, detect_and_draw};

mod mouse_callback;
use crate::mouse_callback::create_mouse_callback;

fn main() -> opencv::Result<()> {
    let (tx, rx) = mpsc::channel();
    let _cam_handle = camera::spawn_camera(tx);

    let overlay = Arc::new(Mutex::new(None::<Mat>));
    highgui::named_window("Screen", highgui::WINDOW_AUTOSIZE)?;

    // mouse callback
    let overlay_cb = overlay.clone();
    let mouse_cb = create_mouse_callback(overlay_cb);

    // SETUP
    let mut frame_count = 0u32;
    let mut last_time = Instant::now();
    let mut current_fps = 0.0;
    let mut camera_enabled = false;
    let mut show_color = true;

    highgui::set_mouse_callback("Screen", mouse_cb)?;

    // draw loop
    loop {
        let mut frame = Mat::default();

        // receive camera frame
        if let Some(cam_rec) = rx.try_recv().ok() {
            frame = cam_rec;
        }

        if frame.empty() {
            continue;
        }

        let frame_clone = frame.clone();
        {
            let mut guard = overlay.lock().unwrap();
            if guard.is_none() {
                *guard = Some(Mat::zeros(frame.rows(), frame.cols(), CV_8UC3)?.to_mat()?);
            }

            // draw based on mode
            if let Some(ref mut overlay_mat) = *guard {
                // draw green
                detect_and_draw(
                    overlay_mat,
                    &frame,
                    &GREEN_RANGE,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                )?;

                // draw blue
                detect_and_draw(
                    overlay_mat,
                    &frame,
                    &BLUE_RANGE,
                    Scalar::new(255.0, 20.0, 20.0, 0.0),
                )?;

                // paint camera to black
                if !camera_enabled {
                    frame.set_to(&Scalar::all(0.0), &Mat::default())?;
                    // draw onto screen
                    overlay_mat.copy_to(&mut frame)?;
                } else {
                    //combine
                    add_weighted(&frame_clone, 1.0, overlay_mat, 1.0, 0.0, &mut frame, -1)?;
                }
            }
        }

        // fps
        frame_count += 1;
        if last_time.elapsed() >= Duration::from_secs(1) {
            current_fps = frame_count as f64 / last_time.elapsed().as_secs_f64();
            frame_count = 0;
            last_time = Instant::now();
        }

        // text to screen
        let text = format!(
            "Resolution: {}x{}\nFPS: {:.1}",
            frame.cols(),
            frame.rows(),
            current_fps,
        );
        imgproc::put_text(
            &mut frame,
            &text,
            Point::new(16, 10),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )
        .ok();

        // show the combined image
        highgui::imshow("Screen", &frame)?;

        let key = highgui::wait_key(10)?;
        // space to toggle camera
        if key == 32 {
            camera_enabled = !camera_enabled;
            continue;
        }
        if key == 118 {
            show_color = !show_color;
            continue;
        }
        // backspace or c to clear
        if key == 8 || key == 99 {
            let mut guard = overlay.lock().unwrap();
            if let Some(ref mut overlay_mat) = *guard {
                overlay_mat.set_to(&Scalar::all(0.0), &Mat::default())?;
            }
            continue;
        }
        // exit on esc or q
        if key == 27 || key == 113 {
            break;
        }
    }

    Ok(())
}
