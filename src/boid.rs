use crate::prelude::*;

struct Coord {
    x: i32,
    y: i32,
}

pub struct Boid {
    position: Coord,
    speed: f64,
    velocity_x: f64,
    velocity_y: f64,
}

impl Boid {
    pub fn new(state: &mut State) -> Self {
        let position = Coord {
            x: state.rng.random_range(0..=state.camera_frame.cols()),
            y: state.rng.random_range(0..=state.camera_frame.rows()),
        };
        let speed = 0.05;
        let velocity_x = state.rng.random_range(-1.0..=1.0);
        let velocity_y = state.rng.random_range(-1.0..=1.0);
        Self {
            position,
            speed,
            velocity_x,
            velocity_y,
        }
    }
    pub fn update(&mut self, state: &mut State, dt: &Duration) {
        
        attract_boid(self, Coord{x: state.camera_frame.cols()/2, y: state.camera_frame.rows()/2});
        random_motion(self, &mut state.rng);

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

fn attract_boid(boid: &mut Boid, pos: Coord) {
    boid.velocity_x += (pos.x - boid.position.x) as f64;
    boid.velocity_y += (pos.y - boid.position.y) as f64;
}

fn random_motion(boid: &mut Boid, rng: &mut ThreadRng) {
    let randomness = 10.0;
    boid.velocity_x += rng.random_range(-randomness..=randomness);
    boid.velocity_y += rng.random_range(-randomness..=randomness);
}


