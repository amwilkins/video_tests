use crate::prelude::*;

// #[derive(Debug)]
// pub struct Coord {
//     pub x: i32,
//     pub y: i32,
// }

#[derive(Debug)]
pub struct Coord {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug)]
pub struct Boid {
    pub position: Coord,
    speed: f64,
    velocity_x: f64,
    velocity_y: f64,
    pub velocity: Coord,
}

impl Boid {
    pub fn new(camera_frame: &Mat, rng: &mut ThreadRng) -> Self {
        let position = Coord {
            x: rng.random_range(0.0..=camera_frame.cols() as f64),
            y: rng.random_range(0.0..=camera_frame.rows() as f64),
        };
        let speed = 0.1;
        let velocity_x = rng.random_range(-1.0..=1.0);
        let velocity_y = rng.random_range(-1.0..=1.0);
        let velocity = Coord {
            x: velocity_x,
            y: velocity_y,
        };
        Self {
            position,
            speed,
            velocity_x,
            velocity_y,
            velocity,
        }
    }
    pub fn update(
        &mut self,
        camera_frame: &Mat,
        rng: &mut ThreadRng,
        centroid: &Coord,
        group_velocity: &Coord,
        //boid_positions,
        dt: &Duration,
    ) {
        let randomness = 100.0;
        random_motion(self, rng.random_range(-randomness..=randomness));

        // center of screen
        attract_boid(
            self,
            &Coord {
                x: camera_frame.cols() as f64 / 2.0,
                y: camera_frame.rows() as f64 / 2.0,
            },
            0.089,
        );

        // centroid
        attract_boid(self, centroid, 0.5);
        
        // align
        //align(self, group_velocity, 0.1);

        // self.velocity.x += (self.velocity.x * self.speed) * dt.as_secs_f64();
        // self.velocity.y += (self.velocity.y * self.speed) * dt.as_secs_f64();

        // self.velocity.x = self.velocity.x * 0.992;
        // self.velocity.y = self.velocity.y * 0.992;

        self.position.x += (self.velocity.x * self.speed) * dt.as_secs_f64();
        self.position.y += (self.velocity.y * self.speed) * dt.as_secs_f64();
    }

    pub fn render(&mut self, overlay: &mut Mat) {
        imgproc::circle(
            overlay,
            Point::new(self.position.x as i32, self.position.y as i32),
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
    boid.velocity.x += (pos.x - boid.position.x) as f64 * str;
    boid.velocity.y += (pos.y - boid.position.y) as f64 * str;
}

fn random_motion(boid: &mut Boid, rng: f64) {
    boid.velocity.x += rng;
    boid.velocity.y += rng;
}

fn align(boid: &mut Boid, vel: &Coord, str: f64) {
    boid.velocity.x = (vel.x + boid.velocity.x) / 2 as f64 * str;
    boid.velocity.y = (vel.y + boid.velocity.y) / 2 as f64 * str;
}

