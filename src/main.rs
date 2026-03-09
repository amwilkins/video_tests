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
    let color_mutex = Arc::new(Mutex::new(None::<Mat>));

    // mouse callback
    let overlay_cb = color_mutex.clone();
    let mouse_cb = create_mouse_callback(overlay_cb);
    highgui::set_mouse_callback("Screen", mouse_cb)?;

    // SETUP
    highgui::named_window("Screen", highgui::WINDOW_AUTOSIZE)?;
    let mut frame_count = 0u32;
    let mut last_time = Instant::now();
    let mut current_fps = 0.0;
    let mut camera_enabled = false;
    let mut show_color = true;

    //creating frames
    let mut camera_frame;
    let mut overlay_frame;
    let mut output_frame;
    let mut color_frame = Mat::default();

    // draw loop
    loop {
        // receive camera frame
        if let Some(cam_rec) = rx.try_recv().ok() {
            camera_frame = cam_rec;
        } else {
            continue;
        }
        // init output frame
        output_frame = Mat::zeros(camera_frame.rows(), camera_frame.cols(), CV_8UC3)?.to_mat()?;

        //let frame_clone = camera_frame.clone();
        {
            let mut guard = color_mutex.lock().unwrap();
            // init frame to cam size
            if guard.is_none() {
                *guard =
                    Some(Mat::zeros(camera_frame.rows(), camera_frame.cols(), CV_8UC3)?.to_mat()?);
            }

            // draw based on mode
            if let Some(ref mut color_mutex) = *guard {
                // draw green
                detect_and_draw(
                    color_mutex,
                    &camera_frame,
                    &GREEN_RANGE,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                )?;

                // draw blue
                detect_and_draw(
                    color_mutex,
                    &camera_frame,
                    &BLUE_RANGE,
                    Scalar::new(255.0, 20.0, 20.0, 0.0),
                )?;
                color_mutex.copy_to(&mut color_frame)?;
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
            camera_frame.cols(),
            camera_frame.rows(),
            current_fps,
        );

        // match overlay frame to output
        overlay_frame = Mat::zeros(camera_frame.rows(), camera_frame.cols(), CV_8UC3)?.to_mat()?;
        imgproc::put_text(
            &mut overlay_frame,
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

        // combining frames to output_frame
        if camera_enabled {
            // add video
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

        // add color frame to output
        add_weighted(
            &output_frame.clone(),
            1.0,
            &color_frame,
            1.0,
            0.0,
            &mut output_frame,
            -1,
        )?;

        // add overlay frame to output
        add_weighted(
            &output_frame.clone(),
            1.0,
            &overlay_frame,
            1.0,
            0.0,
            &mut output_frame,
            -1,
        )?;

        // show the combined image
        highgui::imshow("Screen", &output_frame)?;

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
            let mut color_guard = color_mutex.lock().unwrap();
            if let Some(ref mut overlay_mat) = *color_guard {
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
