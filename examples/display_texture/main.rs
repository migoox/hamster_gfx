use glfw::Context;
use std::time::Instant;
use core::default::Default;
use std::mem::size_of_val;
use std::path::Path;
use hamster_gfx::egui_integration;
use hamster_gfx::renderer::{Shader, ShaderProgram, VertexAttrib, Buffer, VertexBufferLayout};

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

    // EGUI INTEGRATION
    let egui_ctx = egui::Context::default();
    let mut egui_painter = egui_integration::EguiPainter::new(&window);
    let mut egui_input = egui_integration::EguiInputHandler::new(&window);

    let mut sine_shift = 0f32;
    let mut amplitude = 50f32;
    let mut test_str =
        "A text box to write in. Cut, copy, paste commands are available.".to_owned();
    // OPENGL WRAPPER TEST
    use hamster_gfx::renderer::{Shader, ShaderProgram, Buffer, VertexBufferLayout, VertexAttrib, VertexArray, Texture};
    use hamster_gfx::renderer::Bindable;
    let vs = Shader::compile_from_path(&Path::new("resources/examples_shaders/shader.vert"), gl::VERTEX_SHADER).unwrap();
    let fs = Shader::compile_from_path(&Path::new("resources/examples_shaders/shader.frag"), gl::FRAGMENT_SHADER).unwrap();
    let program = ShaderProgram::link(&vs, &fs);

    let mut vao = VertexArray::new();
    let mut vbo_pos = Buffer::new(gl::ARRAY_BUFFER, gl::DYNAMIC_DRAW);
    let mut vbo_tex = Buffer::new(gl::ARRAY_BUFFER, gl::STATIC_DRAW);
    let ebo = Buffer::new(gl::ELEMENT_ARRAY_BUFFER, gl::STATIC_DRAW);

    // Initialize vertex buffers
    let vbo_buff: [f32; 8] = [
        0.5f32, 0.5f32,
        0.5f32, -0.5f32,
        -0.5f32, -0.5f32,
        -0.5f32, 0.5f32,
    ];
    vbo_pos.buffer_data(size_of_val(&vbo_buff), vbo_buff.as_ptr().cast()).unwrap();

    let vbo_buff: [f32; 8] = [
        1.0f32, 1.0f32,
        1.0f32, 0.0f32,
        0.0f32, 0.0f32,
        0.0f32, 1.0f32,
    ];
    vbo_tex.buffer_data(size_of_val(&vbo_buff), vbo_buff.as_ptr().cast()).unwrap();

    // Create vertex buffer layout
    let mut vbl = VertexBufferLayout::new();
    vbl.push_attrib(0 as _, VertexAttrib::new(2, gl::FLOAT, gl::FALSE));

    // Attach position vertex buffer to the layout
    vao.attach_vbo(&vbo_pos, &vbl, 0).unwrap();

    let mut vbl = VertexBufferLayout::new();
    vbl.push_attrib(1 as _, VertexAttrib::new(2, gl::FLOAT, gl::FALSE));

    // Attach texel vertex buffer to the layout
    vao.attach_vbo(&vbo_tex, &vbl, 0).unwrap();

    // Initialize element buffer
    let ebo_buff: [u32; 6] = [
        0, 1, 3,
        1, 2, 3
    ];
    ebo.buffer_data(size_of_val(&ebo_buff), ebo_buff.as_ptr().cast()).unwrap();

    // Create texture
    let mut texture = Texture::new(gl::TEXTURE_2D, gl::LINEAR, gl::CLAMP_TO_EDGE);
    texture.tex_image2d_from_path(&Path::new("resources/images/test_photo.png")).unwrap();


    let mut clock = Instant::now();
    while !window.should_close() {
        // UPDATE INPUT
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Close => window.set_should_close(true),
                _ => {
                    egui_input.handle_event(event);
                }
            }
        }

        // UPDATE
        egui_input.update(&window, clock.elapsed().as_secs_f64());
        egui_painter.update(&window);

        let egui::FullOutput {
            platform_output,
            repaint_after: _,
            textures_delta,
            shapes,
        } = egui_ctx.run(egui_input.take_raw_input(), |ctx| {
            // Egui calls here

            egui::Window::new("Egui with GLFW").show(&egui_ctx, |ui| {
                egui::TopBottomPanel::top("Top").show(&egui_ctx, |ui| {
                    ui.menu_button("File", |ui| {
                        {
                            let _ = ui.button("test 1");
                        }
                        ui.separator();
                        {
                            let _ = ui.button("test 2");
                        }
                    });
                });

                ui.separator();
                ui.label(" ");
                ui.text_edit_multiline(&mut test_str);
                ui.label(" ");
                ui.add(egui::Slider::new(&mut amplitude, 0.0..=50.0).text("Amplitude"));
                ui.label(" ");
                if ui.button("Quit").clicked() {}
            });
        });

        egui_input.handle_clipboard(platform_output);

        // RENDER
        unsafe {
            gl::ClearColor(0.0, 1.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        // Draw a rectangle
        program.bind();
        texture.activate(3);
        program.activate_sampler("u_texture_3", 3).unwrap();
        vao.bind();
        vao.use_vbo(&vbo_pos);
        vao.use_vbo(&vbo_tex);
        vao.use_ebo(&ebo);

        unsafe {
            gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, core::ptr::null());
        }

        // Draw egui content using egui_painter
        egui_painter.paint(&egui_ctx.tessellate(shapes), &textures_delta);

        window.swap_buffers();
        glfw.poll_events();
    }
}
