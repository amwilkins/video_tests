use opencv::{
    Result,
    core::{Mat, Point, Scalar, Vector, find_non_zero, in_range},
    imgproc,
};

pub struct ColorRange {
    pub h_low: i32,
    pub s_low: i32,
    pub v_low: i32,
    pub h_high: i32,
    pub s_high: i32,
    pub v_high: i32,
}

pub fn detect_and_draw(overlay: &mut Mat, frame: &Mat, range: &ColorRange) -> Result<()> {
    // hsv color
    let mut hsv = Mat::default();
    imgproc::cvt_color(frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

    let lower = Scalar::new(
        range.h_low as f64,
        range.s_low as f64,
        range.v_low as f64,
        0.0,
    );
    let upper = Scalar::new(
        range.h_high as f64,
        range.s_high as f64,
        range.v_high as f64,
        0.0,
    );
    let mut mask = Mat::default();
    in_range(&hsv, &lower, &upper, &mut mask)?;

    // find points that are lit up and draw to screen
    let mut points: Vector<Point> = Vector::new();
    find_non_zero(&mask, &mut points)?;
    for p in points.iter() {
        imgproc::circle(
            overlay,
            Point::new(p.x, p.y),
            2,
            Scalar::new(0.0, 255.0, 0.0, 0.0), // green for now
            -1,
            imgproc::LINE_8,
            0,
        )?;
    }
    Ok(())
}
