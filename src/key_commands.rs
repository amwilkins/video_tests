use crate::prelude::*;

enum KeyCode {
    Space,
    C,
    V,
    Backspace,
    Escape,
    Q,
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
        _ => KeyCode::None
    }
}

fn clear_color_overlay(state: &mut State){
    let mut color_guard = state.color_overlay.overlay.lock().unwrap();
    if let Some(ref mut overlay_mat) = *color_guard {
        overlay_mat.set_to(&Scalar::all(0.0), &Mat::default()).unwrap();
    }
}

pub fn handle_keypress(key: i32, state: &mut State){
    let keycode = get_keycode(key);

    match keycode {
        KeyCode::Space => state.camera_enabled = !state.camera_enabled,
        KeyCode::V => state.reset_color_mode = !state.reset_color_mode,
        KeyCode::C => clear_color_overlay(state),
        KeyCode::Q => std::process::exit(0),
        KeyCode::Escape => std::process::exit(0),
        _ => ()
    }

}
