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
    pub color: Scalar,
}

// Common color ranges
pub const GREEN_RANGE: ColorRange = ColorRange {
    h_low: 40,
    s_low: 60,
    v_low: 120,
    h_high: 75,
    s_high: 255,
    v_high: 255,
    color: Scalar::new(0.0, 255.0, 0.0, 0.0),
};

pub const BLUE_RANGE: ColorRange = ColorRange {
    h_low: 90,
    s_low: 190,
    v_low: 190,
    h_high: 115,
    s_high: 255,
    v_high: 240,
    color: Scalar::new(255.0, 10.0, 10.0, 0.0),
};

pub const RED_RANGE: ColorRange = ColorRange {
    h_low: 150,
    s_low: 180,
    v_low: 180,
    h_high: 180,
    s_high: 255,
    v_high: 240,
    color: Scalar::new(10.0, 10.0, 255.0, 0.0),
};

pub fn detect_and_draw(
    overlay: &mut Mat,
    frame: &Mat,
    detect_colors: &Vec<&ColorRange>,
) -> Result<()> {
    // convert to hsv color
    let mut hsv = Mat::default();
    imgproc::cvt_color(frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

    for range in detect_colors {
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
                range.color,
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }
    }
    Ok(())
}
