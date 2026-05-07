use crate::prelude::*;

#[derive(Debug)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug)]
pub struct Boid {
    pub position: Coord,
    speed: f64,
    velocity_x: f64,
    velocity_y: f64,
}

impl Boid {
    pub fn new(camera_frame: &Mat, rng: &mut ThreadRng) -> Self {
        let position = Coord {
            x: rng.random_range(0..=camera_frame.cols()),
            y: rng.random_range(0..=camera_frame.rows()),
        };
        let speed = 0.05;
        let velocity_x = rng.random_range(-1.0..=1.0);
        let velocity_y = rng.random_range(-1.0..=1.0);
        Self {
            position,
            speed,
            velocity_x,
            velocity_y,
        }
    }
    pub fn update(&mut self, camera_frame: &Mat, rng: &mut ThreadRng, centroid: &Coord, dt: &Duration) {
        let randomness = 100.0;
        random_motion(self, rng.random_range(-randomness..=randomness));
        
        // center of screen
        attract_boid(self, &Coord{x: camera_frame.cols()/2, y: camera_frame.rows()/2}, 0.02);

        // centroid
        attract_boid(self, centroid, 0.8);

        self.position.x += ((self.velocity_x * self.speed) * dt.as_secs_f64()) as i32;
        self.position.y += ((self.velocity_y * self.speed) * dt.as_secs_f64()) as i32;
    }

    pub fn render(&mut self, overlay: &mut Mat) {
        imgproc::circle(
            overlay,
            Point::new(self.position.x, self.position.y),
            5,
            Scalar::all(255.0),
            -1,
            imgproc::LINE_8,
            0,
        )
        .ok();
    }
}

fn attract_boid(boid: &mut Boid, pos: &Coord, str: f64) {
    boid.velocity_x += (pos.x - boid.position.x) as f64 * str;
    boid.velocity_y += (pos.y - boid.position.y) as f64 * str;
}

fn random_motion(boid: &mut Boid, rng: f64) {
    boid.velocity_x += rng;
    boid.velocity_y += rng;
}


