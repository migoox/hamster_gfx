use glfw::Context;
use std::time::Instant;
use core::default::Default;
use std::mem::size_of_val;
use std::path::Path;
use egui::Color32;
use hamster_gfx::egui_integration;
use hamster_gfx::egui_integration::EguiUserTexture;
use hamster_gfx::renderer::{Shader, ShaderProgram, VertexAttrib, Buffer, VertexBufferLayout, Bindable};

const SCREEN_WIDTH: u32 = 1600;
const SCREEN_HEIGHT: u32 = 1200;

const SIN_PIC_WIDTH: usize = 200;
const SIN_PIC_HEIGHT: usize = 200;

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
    let mut egui_io = egui_integration::EguiIOHandler::new(&window);

    // Egui Textures examples
    let mut gl_texture = Texture::new(gl::TEXTURE_2D, gl::LINEAR, gl::CLAMP_TO_EDGE);
    gl_texture.tex_image2d_from_path_no_flip(&Path::new("resources/images/hamster2.png")).unwrap();
    let egui_txt = EguiUserTexture::from_gl_texture(&mut egui_painter, gl_texture, false).unwrap();

    let egui_sin_txt = EguiUserTexture::new(
        &mut egui_painter,
        egui::TextureFilter::Linear,
        SIN_PIC_WIDTH,
        SIN_PIC_HEIGHT,
        false,
        &vec![Color32::from_rgb(15, 15, 15); SIN_PIC_WIDTH * SIN_PIC_HEIGHT],
    );

    let mut sin_data: Vec<Color32> = vec![Color32::BLACK; SIN_PIC_HEIGHT * SIN_PIC_WIDTH];

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
    texture.tex_image2d_from_path(&Path::new("resources/images/hamster.png")).unwrap();

    program.activate_sampler("u_texture", 3).unwrap();

    let mut clock = Instant::now();

    while !window.should_close() {
        // UPDATE INPUT
        for (_, event) in glfw::flush_messages(&events) {
            if event == glfw::WindowEvent::Close {
                window.should_close();
            } else {
                match event {
                    // Custom event handling
                    _ => (),
                };

                // Move GLFW events as an input into EguiIOHandler
                egui_io.handle_event(event);
            }
        }

        {
            // SINUS TEXTURE UPDATE

            for x in 0..SIN_PIC_WIDTH {
                for y in 0..SIN_PIC_HEIGHT {
                    sin_data[(y as i32 * (SIN_PIC_WIDTH as i32) + (x as i32)) as usize] = Color32::BLACK;
                }
                // get y position for x
                let y = amplitude * ((x as f32) * std::f32::consts::PI / 180f32 + sine_shift).sin();
                let y = SIN_PIC_HEIGHT as f32 / 2f32 - y;
                sin_data[(y as i32 * (SIN_PIC_WIDTH as i32) + (x as i32)) as usize] = Color32::YELLOW;
            }

            // update sinus shift so that it "moves" in each frame
            sine_shift += 0.05f32;

            egui_sin_txt.update(&sin_data);
        }

        // START AN EGUI FRAME (it should happen before egui integration update)
        egui_ctx.begin_frame(egui_io.take_raw_input());

        // UPDATE EGUI INTEGRATION
        egui_io.update(&window, clock.elapsed().as_secs_f64());
        egui_painter.update(&window);

        // Egui calls
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

            ui.add(egui::Image::new(egui_txt.get_id(), egui_txt.get_size()));
            ui.separator();
            ui.label(" ");
            ui.text_edit_multiline(&mut test_str);
            ui.label(" ");
            ui.add(egui::Slider::new(&mut amplitude, 0.0..=50.0).text("Amplitude"));
            ui.label(" ");
            ui.add(egui::Image::new(egui_sin_txt.get_id(), egui_sin_txt.get_size()));
        });


        // END AN EGUI FRAME
        let egui::FullOutput {
            platform_output,
            repaint_after: _,
            textures_delta,
            shapes,
        } = egui_ctx.end_frame();

        egui_io.handle_platform_output(platform_output, &mut window);

        // RENDER
        unsafe {
            gl::ClearColor(0.0, 1.0, 0.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        // Draw a rectangle
        vao.bind();
        program.bind();
        texture.activate(3);
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
