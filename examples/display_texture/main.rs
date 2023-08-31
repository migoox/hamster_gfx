use glfw::Context;
use std::time::Instant;
use core::default::Default;
use std::mem::size_of_val;
use std::path::Path;
use hamster_rustygfx::renderer::{Shader, ShaderProgram, VertexAttrib, Buffer, VertexBufferLayout};

const SCREEN_WIDTH: u32 = 1600;
const SCREEN_HEIGHT: u32 = 1200;

fn main() {
    // TO DO: callback for errors
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    // Set window hints
    glfw.window_hint(glfw::WindowHint::Resizable(true));

    // Create GLFW window
    let (mut window, events) = glfw
        .create_window(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            "Window",
            glfw::WindowMode::Windowed,
        )
        .expect("Failed to create the GLFW window");

    // Enable detecting of text writing event
    window.set_char_polling(true);
    // This function sets the cursor position callback of the specified window,
    // which is called when the cursor is moved
    window.set_cursor_pos_polling(true);
    // This function sets the key callback of the specified window,
    // which is called when a key is pressed, repeated or released.
    window.set_key_polling(true);
    // Mouse button callback
    window.set_mouse_button_polling(true);
    // Make context of the window current (attach it to the current thread)
    window.make_current();

    // Turn on VSync (window_refresh_interval:screen_refresh_interval - 1:1)
    glfw.set_swap_interval(glfw::SwapInterval::Sync(1));

    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

    let native_pixels_per_point = window.get_content_scale().0;

    // Create egui_shaders context
    let egui_ctx = egui::Context::default();

    let mut clock: Instant;

    // OPENGL WRAPPER TEST
    use hamster_rustygfx::renderer::{Shader, ShaderProgram, Buffer, VertexBufferLayout, VertexAttrib, VertexArray, Texture};
    use hamster_rustygfx::renderer::Bindable;
    let vs = Shader::compile_from_path(&Path::new("resources/examples_shaders/shader.vert"), gl::VERTEX_SHADER).unwrap();
    let fs = Shader::compile_from_path(&Path::new("resources/examples_shaders/shader.frag"), gl::FRAGMENT_SHADER).unwrap();
    let program = ShaderProgram::link(&vs, &fs);
    program.bind();

    let mut vao = VertexArray::new();
    let mut vbo = Buffer::new(gl::ARRAY_BUFFER, gl::STATIC_DRAW);
    let ebo = Buffer::new(gl::ELEMENT_ARRAY_BUFFER, gl::STATIC_DRAW);

    // Initialize vertex buffer
    let vbo_buff: [f32; 16] = [
        0.5f32, 0.5f32, 1.0f32, 1.0f32,
        0.5f32, -0.5f32, 1.0f32, 0.0f32,
        -0.5f32, -0.5f32, 0.0f32, 0.0f32,
        -0.5f32, 0.5f32, 0.0f32, 1.0f32,
    ];
    vbo.buffer_data(size_of_val(&vbo_buff) as u32, vbo_buff.as_ptr().cast()).unwrap();

    // Create vertex buffer layout
    let mut vbl = VertexBufferLayout::new();
    vbl.push_attrib(0 as _, VertexAttrib::new(2, gl::FLOAT, gl::FALSE))
        .push_attrib(1 as _, VertexAttrib::new(2, gl::FLOAT, gl::FALSE));

    // Attach vertex buffer to the layout
    vao.attach_vbo(&vbo, &vbl, 0).unwrap();

    // Initialize element buffer
    let ebo_buff: [u32; 6] = [
        0, 1, 3,
        1, 2, 3
    ];
    ebo.buffer_data(size_of_val(&ebo_buff) as u32, ebo_buff.as_ptr().cast()).unwrap();

    // Create texture
    let mut texture = Texture::new(gl::TEXTURE_2D, gl::LINEAR, gl::CLAMP_TO_EDGE);
    texture.tex_image2d_from_path(&Path::new("resources/images/test_photo.png")).unwrap();
    texture.activate(3);
    program.activate_sampler("u_texture_3", 3).unwrap();

    while !window.should_close() {
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Close => window.set_should_close(true),
                _ => {
                    // Translate GLFW events to egui_shaders events
                }
            }
        }

        clock = Instant::now();

        // Get size in pixels
        let (width, height) = window.get_framebuffer_size();

        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::new(0f32, 0f32),
                egui::vec2(width as f32, height as f32) / native_pixels_per_point,
            )),
            ..Default::default()
        };

        egui_ctx.begin_frame(raw_input);

        let egui::FullOutput {
            platform_output: _platform_output,
            repaint_after: _,
            textures_delta: _textures_delta,
            shapes,
        } = egui_ctx.end_frame();

        let _clipped_primitives = egui_ctx.tessellate(shapes);

        // Clear here
        unsafe {
            gl::ClearColor(0.0, 1.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        // Draw a rectangle
        program.bind();
        vao.bind();
        vao.use_vbo(&vbo);
        vao.use_ebo(&ebo);

        unsafe {
            // gl::DrawArrays(gl::TRIANGLES, 0, 3);
            gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, core::ptr::null());
        }

        // Draw clipped primitives

        // Draw here

        window.swap_buffers();
        glfw.poll_events();
    }
}
