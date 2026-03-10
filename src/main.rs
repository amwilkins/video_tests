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
use crate::color_detect::*;

mod mouse_callback;
use crate::mouse_callback::create_mouse_callback;

fn main() -> opencv::Result<()> {
    let (tx, rx) = mpsc::channel();
    let _cam_handle = camera::spawn_camera(tx);
    let color_mutex = Arc::new(Mutex::new(None::<Mat>));
    highgui::named_window("Screen", highgui::WINDOW_AUTOSIZE)?;

    // ~~~ SETUP ~~~ //
    let mut frame_count = 0u32;
    let mut last_time = Instant::now();
    let mut current_fps = 0.0;

    // flags
    let mut camera_enabled = false;
    //let mut show_color = true;
    let mut reset_color_mode = false;

    // mouse callback
    let mouse_cb = create_mouse_callback(color_mutex.clone());
    highgui::set_mouse_callback("Screen", mouse_cb)?;

    // select detection colors
    let detect_colors = vec![&GREEN_RANGE, &BLUE_RANGE, &RED1_RANGE, &RED2_RANGE];

    //creating frames
    let mut color_frame = Mat::default();
    let mut camera_frame = rx.recv().unwrap(); // blocking wait for first frame
    let mut output_frame =
        Mat::zeros(camera_frame.rows(), camera_frame.cols(), CV_8UC3)?.to_mat()?;

    let mut overlay_frame =
        Mat::zeros(camera_frame.rows(), camera_frame.cols(), CV_8UC3)?.to_mat()?;
    // ~~~ draw loop ~~~ //
    loop {
        // receive camera frame
        if let Some(cam_rec) = rx.try_recv().ok() {
            camera_frame = cam_rec;
        } else {
            continue;
        }

        {
            let mut guard = color_mutex.lock().unwrap();
            // init frame to cam size
            if guard.is_none() {
                *guard =
                    Some(Mat::zeros(camera_frame.rows(), camera_frame.cols(), CV_8UC3)?.to_mat()?);
            }

            // detect and draw
            if let Some(ref mut color_mutex) = *guard {
                // draw based on mode
                if reset_color_mode {
                    color_mutex.set_to(&Scalar::all(0.0), &Mat::default())?;
                }

                detect_and_draw(color_mutex, &camera_frame, &detect_colors)?;
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

        // add overlay frame to output
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
        if camera_enabled {
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
        // space to toggle camera
        if key == 32 {
            camera_enabled = !camera_enabled;
            continue;
        }
        if key == 118 {
            //show_color = !show_color;
            println!("Reset color mode");
            reset_color_mode = !reset_color_mode;
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
