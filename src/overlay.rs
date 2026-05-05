use crate::prelude::*;



pub fn create_overlay(state: &State) -> Mat {
    let mut overlay_frame = Mat::zeros(state.camera_rows, state.camera_cols, CV_8UC3).unwrap().to_mat();
    overlay_frame.unwrap()
}
