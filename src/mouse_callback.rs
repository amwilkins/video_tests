use opencv::{
    core::{Mat, Point, Scalar},
    highgui, imgproc,
    prelude::*,
};
use std::sync::{Arc, Mutex};

pub fn create_mouse_callback(
    overlay: Arc<Mutex<Option<Mat>>>,
) -> Option<Box<dyn FnMut(i32, i32, i32, i32) + Send + Sync + 'static>> {
    let mouse_cb: highgui::MouseCallback = Some(Box::new(move |event, x, y, flags| {
        let mut guard = overlay.lock().unwrap();

        // match mouse buttons
        if let Some(ref mut img) = *guard {
            // left button to erase
            if event == highgui::EVENT_LBUTTONDOWN
                || (event == highgui::EVENT_MOUSEMOVE && (flags & highgui::EVENT_FLAG_LBUTTON) != 0)
            {
                imgproc::circle(
                    img,
                    Point::new(x, y),
                    20,
                    Scalar::all(0.0),
                    -1,
                    imgproc::LINE_8,
                    0,
                )
                .ok();
            }
            // right or middle button to clear
            else if event == highgui::EVENT_RBUTTONDOWN || event == highgui::EVENT_MBUTTONDOWN {
                img.set_to(&Scalar::all(0.0), &Mat::default()).ok();
            }
        }
    }));
    return mouse_cb;
}
