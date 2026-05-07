use crate::prelude::*;

enum KeyCode {
    Space,
    c,
    v,
    Backspace,
    Escape,
    q,
    r,
    None,
}

fn get_keycode(key: i32) -> KeyCode {
    return match key {
        32 => KeyCode::Space,
        118 => KeyCode::v,
        99 => KeyCode::c,
        8 => KeyCode::Backspace,
        27 => KeyCode::Escape,
        113 => KeyCode::q,
        114 => KeyCode::r,
        _ => KeyCode::None,
    };
}

pub fn handle_keypress(key: i32, state: &mut State) {
    let keycode = get_keycode(key);

    match keycode {
        KeyCode::Space => state.camera_enabled = !state.camera_enabled,
        KeyCode::v => state.reset_color_mode = !state.reset_color_mode,
        KeyCode::c => clear_color_overlay(state),
        KeyCode::q => std::process::exit(0),
        //KeyCode::r => state = State::new(state.camera_frame),
        KeyCode::Escape => std::process::exit(0),
        _ => (),
    }
}

