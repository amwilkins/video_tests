use crate::prelude::*;

enum KeyCode {
    Space,
    C,
    V,
    Backspace,
    Escape,
    Q,
    R,
    None,
}

fn get_keycode(key: i32) -> KeyCode {
    return match key {
        32 => KeyCode::Space,
        118 => KeyCode::V,
        99 => KeyCode::C,
        8 => KeyCode::Backspace,
        27 => KeyCode::Escape,
        113 => KeyCode::Q,
        114 => KeyCode::R,
        _ => KeyCode::None,
    };
}

pub fn handle_keypress(key: i32, state: &mut State) {
    let keycode = get_keycode(key);

    match keycode {
        KeyCode::Space => state.camera_enabled = !state.camera_enabled,
        KeyCode::V => state.reset_color_mode = !state.reset_color_mode,
        KeyCode::C => clear_color_overlay(state),
        KeyCode::Q => std::process::exit(0),
        //KeyCode::R => state = State::new(state.camera_frame),
        KeyCode::Escape => std::process::exit(0),
        _ => (),
    }
}

