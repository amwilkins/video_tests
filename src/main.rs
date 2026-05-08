use opencv::{
    //core::{add_weighted, Mat, Point, Scalar, CV_8UC3},
    highgui,
    imgproc,
};
use rand::RngExt;
use std::sync::mpsc;
use std::time::{Duration, Instant};

mod boid;
mod camera;
mod color_detect;
mod key_commands;
//mod overlay;
//mod mouse_callback;

mod prelude {
    pub use crate::boid::*;
    pub use crate::camera::*;
    pub use crate::color_detect::*;
    pub use crate::key_commands::*;
    //pub use crate::overlay::*;
    pub use crate::*;
    //pub use crate::mouse_callback::create_mouse_callback;
    pub use opencv::core::*;
    pub use opencv::prelude::*;
    pub use rand::rngs::ThreadRng;
}

use prelude::*;

const NUM_BOIDS: i32 = 5;

#[derive(Debug)]
pub struct State {
    camera_enabled: bool,
    reset_color_mode: bool,
    color_overlay: ColorOverlay,
    //output_frame: Mat,
    camera_frame: Mat,
    rng: ThreadRng,
    boids: Vec<Boid>,
}

impl State {
    fn new(camera_frame: Mat) -> Self {
        let mut rng = rand::rng();
        let mut boid_vec: Vec<Boid> = Vec::new();

        for _ in 0..=NUM_BOIDS {
            boid_vec.push(Boid::new(&camera_frame, &mut rng))
        }

        Self {
            camera_enabled: false,
            reset_color_mode: false,
            color_overlay: ColorOverlay::new(),
            //output_frame: Mat::default(),
            camera_frame,
            rng: rng,
            boids: boid_vec,
        }
    }
}

fn main() -> opencv::Result<()> {
    let (tx, rx) = mpsc::channel();
    let _cam_handle = spawn_camera(tx);
    highgui::named_window("Screen", highgui::WINDOW_KEEPRATIO)?;
    highgui::resize_window("Screen", 1600, 900)?;

    //highgui::named_window("Screen", highgui::WINDOW_AUTOSIZE)?;

    let camera_frame = rx.recv().unwrap(); // blocking wait for first frame
    let mut state = State::new(camera_frame);

    // ~~~ FPS SETUP ~~~ //
    let mut frame_count = 0u32;
    let mut last_time = Instant::now();
    let mut current_fps = 0.0;

    // // mouse callback
    // let mouse_cb = create_mouse_callback(color_mutex.clone());
    // highgui::set_mouse_callback("Screen", mouse_cb)?;

    // select detection colors
    let detect_colors = vec![&GREEN_RANGE, &BLUE_RANGE, &RED1_RANGE, &RED2_RANGE];

    // init frames
    let mut output_frame = Mat::zeros(
        state.camera_frame.rows(),
        state.camera_frame.cols(),
        CV_8UC3,
    )?
    .to_mat()?;
    let mut overlay_frame = Mat::zeros(
        state.camera_frame.rows(),
        state.camera_frame.cols(),
        CV_8UC3,
    )?
    .to_mat()?;
    state.color_overlay.overlay = Mat::zeros(
        state.camera_frame.rows(),
        state.camera_frame.cols(),
        CV_8UC3,
    )?
    .to_mat()?;

    // init time
    let dt = Duration::new(1, 0) / 60;
    let mut t = Duration::new(0, 0);
    let mut current_time = Instant::now();
    let mut accumulator = Duration::new(0, 0);

    // ~~~ draw loop ~~~ //
    loop {
        // time //
        let newtime = Instant::now();
        let frametime = newtime - current_time;
        current_time = newtime;
        accumulator += frametime;

        // Camera frame, non-blocking //
        if let Some(cam_rec) = rx.try_recv().ok() {
            // mirror x axis
            flip(&cam_rec.clone(), &mut state.camera_frame, 1)?;

            // fps
            frame_count += 1;
            if last_time.elapsed() >= Duration::from_secs(1) {
                current_fps = frame_count as f64 / last_time.elapsed().as_secs_f64();
                frame_count = 0;
                last_time = Instant::now();
            }
            // Detect and draw color
            if state.reset_color_mode {
                clear_color_overlay(&mut state);
            }
            detect_and_draw_color(
                &mut state.color_overlay.overlay,
                &state.camera_frame.clone(),
                &detect_colors,
            )?;
        }

        // TODO: create overlay frame
        // overlay_frame = create_overlay(&state, current_fps);

        // text to screen
        let text = format!(
            "Resolution: {}x{}\nFPS: {:.1}",
            state.camera_frame.cols(),
            state.camera_frame.rows(),
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

        // add color_overlay to output
        add_weighted(
            &state.color_overlay.overlay,
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
                &state.camera_frame.clone(),
                1.0,
                0.0,
                &mut output_frame,
                -1,
            )?;
        }

        // render boids
        while accumulator >= dt {

            let mut boid_centeroid = Coord { x: 0.0, y: 0.0 };
            let mut boid_velocity = Coord { x: 0.0, y: 0.0 };
            for b in state.boids.iter().clone() {
                boid_centeroid.x += b.position.x;
                boid_centeroid.y += b.position.y;
                boid_velocity.x += b.velocity.x;
                boid_velocity.y += b.velocity.y;
            }
            boid_centeroid.x /= state.boids.len() as f64;
            boid_centeroid.y /= state.boids.len() as f64;

            boid_velocity.x /= state.boids.len() as f64;
            boid_velocity.y /= state.boids.len() as f64;

            //let boid_positions = state.boids.iter().collect();

            for boid in &mut state.boids {
                boid.update(&state.camera_frame, &mut state.rng, &boid_centeroid, &boid_velocity, &dt);
            }

            accumulator -= dt;
            t += dt;
        }
        for boid in &mut state.boids {
            boid.render(&mut output_frame);
        }

        // show the combined image
        highgui::imshow("Screen", &output_frame)?;

        let key = highgui::wait_key(1)?;
        handle_keypress(key, &mut state);
    }
}
