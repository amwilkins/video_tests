use crate::prelude::*;

pub fn create_overlay(state: &State, current_fps: f64) -> Mat {
    let mut overlay_frame = Mat::zeros(state.camera_rows, state.camera_cols, CV_8UC3)
        .unwrap()
        .to_mat()
        .unwrap();


    // text to screen
    let text = format!(
        "Resolution: {}x{}\nFPS: {:.1}",
        state.camera_rows, state.camera_cols, current_fps,
    );

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

    overlay_frame
}
