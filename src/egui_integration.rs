use std::path::Path;
use core::default::Default;
use cli_clipboard::{ClipboardContext, ClipboardProvider};
use egui::*;
use gl::types::{GLsizei, GLvoid};
use crate::renderer::{Shader, ShaderProgram, Buffer, VertexArray, Bindable, VertexBufferLayout, VertexAttrib};

/// Egui Painter uses 0th texture unit. If you are going to use 0th
/// texture unit, remember to activate it once per draw call.
pub struct EguiPainter {
    shader_program: ShaderProgram,

    vao: VertexArray,
    vbo: Buffer,
    ebo: Buffer,

    canvas_width: i32,
    canvas_height: i32,
    native_pixels_per_point: f32,
}

impl EguiPainter {
    pub fn new(glfw_window: &glfw::Window) -> EguiPainter {
        let vs = Shader::compile_from_path(&Path::new("resources/egui_shaders/vertex.vert"), gl::VERTEX_SHADER)
            .expect("Painter couldn't load the vertex egui_shader");
        let fs = Shader::compile_from_path(&Path::new("resources/egui_shaders/fragment.frag"), gl::FRAGMENT_SHADER)
            .expect("Painter couldn't load the fragment egui_shader");

        let shader_program = ShaderProgram::link(&vs, &fs);

        let mut vao = VertexArray::new();
        let mut vbo = Buffer::new(gl::ARRAY_BUFFER, gl::STREAM_DRAW);
        let mut vbl = VertexBufferLayout::new();
        vbl.push_attrib(0, VertexAttrib::new(2, gl::FLOAT, gl::FALSE)) //
            .push_attrib(1, VertexAttrib::new(4, gl::UNSIGNED_BYTE, gl::FALSE))
            .push_attrib(2, VertexAttrib::new(2, gl::FLOAT, gl::FALSE));

        vao.attach_vbo(&vbo, &vbl, 0).unwrap();

        let mut result = EguiPainter {
            shader_program,
            vao,
            vbo,
            ebo: Buffer::new(gl::ELEMENT_ARRAY_BUFFER, gl::STREAM_DRAW),
            canvas_width: 0,
            canvas_height: 0,
            native_pixels_per_point: 1.0,
        };
        result.update(&glfw_window);
        result
    }

    pub fn update(&mut self, glfw_window: &glfw::Window) {
        (self.canvas_width, self.canvas_height) = glfw_window.get_framebuffer_size();
        self.native_pixels_per_point = glfw_window.get_content_scale().0;
    }

    pub fn paint(&mut self, clipped_primitives: Vec<ClippedPrimitive>, texture_delta: TexturesDelta) {

        // Prepare OpenGL
        unsafe {
            // Let OpenGL know we are dealing with SRGB colors so that it
            // can do the blending correctly. Not setting the framebuffer
            // leads to darkened, oversaturated colors.
            gl::Enable(gl::FRAMEBUFFER_SRGB);
            gl::Enable(gl::SCISSOR_TEST);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::ONE, gl::ONE_MINUS_SRC_ALPHA); // premultiplied alpha

            // Specify viewport
            gl::Viewport(0, 0, self.canvas_width, self.canvas_height);
        }

        // Prepare Uniforms
        self.shader_program.set_uniform2f(
            "u_screen_size",
            self.canvas_width as f32 / self.native_pixels_per_point,
            self.canvas_height as f32 / self.native_pixels_per_point
        );

        self.shader_program.activate_sampler("u_sampler", 0).unwrap();

        // Iterate through the clipped primitives produced by egui
        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_primitives
        {
            match primitive {
                epaint::Primitive::Mesh(mesh) => {
                    self.paint_mesh(mesh, clip_rect);
                }

                epaint::Primitive::Callback(_) => {
                    panic!("Custom rendering callbacks are not implemented");
                }
            }
        }

        // Clean OpenGL state
        unsafe {
            // Let OpenGL know we are dealing with SRGB colors so that it
            // can do the blending correctly. Not setting the framebuffer
            // leads to darkened, oversaturated colors.
            gl::Disable(gl::FRAMEBUFFER_SRGB);
            gl::Disable(gl::SCISSOR_TEST);
            gl::Disable(gl::BLEND);
        }
    }

    pub fn paint_mesh(&self, mesh: Mesh, clip_rect: Rect) {
        debug_assert!(mesh.is_valid());

        // BeginFrom: https://github.com/emilk/egui/blob/master/crates/egui_glium/src/painter.rs
        // Transform clip rect to physical pixels:
        let clip_min_x = self.native_pixels_per_point * clip_rect.min.x;
        let clip_min_y = self.native_pixels_per_point * clip_rect.min.y;
        let clip_max_x = self.native_pixels_per_point * clip_rect.max.x;
        let clip_max_y = self.native_pixels_per_point * clip_rect.max.y;

        // Make sure clip rect can fit within a `u32`:
        let clip_min_x = clip_min_x.clamp(0.0, self.canvas_width as f32);
        let clip_min_y = clip_min_y.clamp(0.0, self.canvas_height as f32);
        let clip_max_x = clip_max_x.clamp(clip_min_x, self.canvas_width as f32);
        let clip_max_y = clip_max_y.clamp(clip_min_y, self.canvas_height as f32);

        let clip_min_x = clip_min_x.round() as GLsizei;
        let clip_min_y = clip_min_y.round() as GLsizei;
        let clip_max_x = clip_max_x.round() as GLsizei;
        let clip_max_y = clip_max_y.round() as GLsizei;
        // EndFrom

        // Perform a scissor test
        unsafe {
            gl::Scissor(
                clip_min_x,
                self.canvas_height - clip_max_y, // Revert y axis
                clip_max_x - clip_min_x,
                clip_max_y - clip_min_y,
            )
        }

        // VAO
        self.vao.bind();

        // Bind appropriate texture
        // here

        self.ebo.buffer_data(
            mesh.indices.len() * core::mem::size_of::<u32>(),
            mesh.indices.as_ptr() as *const GLvoid,
        ).unwrap();

        self.vbo.buffer_data(
            mesh.vertices.len() *
                (2 * core::mem::size_of::<f32>()        // position
                    + 4 * core::mem::size_of::<u8>()    // color
                    + 2 * core::mem::size_of::<f32>()   // texels
                ),
            mesh.vertices.as_ptr() as *const GLvoid,
        ).unwrap();

        self.shader_program.bind();
        self.vao.bind();
        self.vao.use_vbo(&self.vbo);
        self.vao.use_ebo(&self.ebo);
        unsafe {
            gl::DrawElements(
                gl::TRIANGLES,
                mesh.indices.len() as _,
                gl::UNSIGNED_SHORT,
                core::ptr::null()
            );
        }
    }
}

// from: https://github.com/cohaereo/egui_glfw_gl
pub struct EguiInputHandler {
    pointer_pos: Pos2,
    clipboard: Option<ClipboardContext>,
    input: RawInput,
    modifiers: Modifiers,
}

impl EguiInputHandler {
    pub fn new(glfw_window: &glfw::Window) -> Self {
        let (width, height) = glfw_window.get_framebuffer_size();
        EguiInputHandler {
            pointer_pos: Pos2::new(0f32, 0f32),
            clipboard: Self::init_clipboard(),
            input: RawInput {
                screen_rect: Some(Rect::from_min_size(
                    Pos2::new(0f32, 0f32),
                    vec2(width as f32, height as f32) / glfw_window.get_content_scale().0,
                )),
                ..Default::default()
            },
            modifiers: Default::default(),
        }
    }

    pub fn handle_event(&mut self, event: glfw::WindowEvent) {
        use glfw::WindowEvent::*;

        match event {
            FramebufferSize(width, height) => {
                self.input.screen_rect = Some(Rect::from_min_size(
                    Pos2::new(0f32, 0f32),
                    egui::vec2(width as f32, height as f32)
                        / self.input.pixels_per_point.unwrap_or(1.0),
                ));
            }

            MouseButton(mouse_btn, glfw::Action::Press, _) => {
                self.input.events.push(egui::Event::PointerButton {
                    pos: self.pointer_pos,
                    button: match mouse_btn {
                        glfw::MouseButtonLeft => egui::PointerButton::Primary,
                        glfw::MouseButtonRight => egui::PointerButton::Secondary,
                        glfw::MouseButtonMiddle => egui::PointerButton::Middle,
                        _ => unreachable!(),
                    },
                    pressed: true,
                    modifiers: self.modifiers,
                })
            }

            MouseButton(mouse_btn, glfw::Action::Release, _) => {
                self.input.events.push(egui::Event::PointerButton {
                    pos: self.pointer_pos,
                    button: match mouse_btn {
                        glfw::MouseButtonLeft => egui::PointerButton::Primary,
                        glfw::MouseButtonRight => egui::PointerButton::Secondary,
                        glfw::MouseButtonMiddle => egui::PointerButton::Middle,
                        _ => unreachable!(),
                    },
                    pressed: false,
                    modifiers: self.modifiers,
                })
            }

            CursorPos(x, y) => {
                self.pointer_pos = pos2(
                    x as f32 / self.input.pixels_per_point.unwrap_or(1.0),
                    y as f32 / self.input.pixels_per_point.unwrap_or(1.0),
                );
                self
                    .input
                    .events
                    .push(egui::Event::PointerMoved(self.pointer_pos))
            }

            Key(keycode, _scancode, glfw::Action::Release, keymod) => {
                use glfw::Modifiers as Mod;
                if let Some(key) = Self::translate_glfwkey_to_eguikey(keycode) {
                    self.modifiers = Modifiers {
                        alt: (keymod & Mod::Alt == Mod::Alt),
                        ctrl: (keymod & Mod::Control == Mod::Control),
                        shift: (keymod & Mod::Shift == Mod::Shift),

                        // TODO: GLFW doesn't seem to support the mac command key
                        //       mac_cmd: keymod & Mod::LGUIMOD == Mod::LGUIMOD,
                        command: (keymod & Mod::Control == Mod::Control),

                        ..Default::default()
                    };

                    self.input.events.push(Event::Key {
                        key,
                        pressed: false,
                        repeat: false,
                        modifiers: self.modifiers,
                    });
                }
            }

            Key(keycode, _scancode, glfw::Action::Press | glfw::Action::Repeat, keymod) => {
                use glfw::Modifiers as Mod;
                if let Some(key) = Self::translate_glfwkey_to_eguikey(keycode) {
                    self.modifiers = Modifiers {
                        alt: (keymod & Mod::Alt == Mod::Alt),
                        ctrl: (keymod & Mod::Control == Mod::Control),
                        shift: (keymod & Mod::Shift == Mod::Shift),

                        // TODO: GLFW doesn't seem to support the mac command key
                        //       mac_cmd: keymod & Mod::LGUIMOD == Mod::LGUIMOD,
                        command: (keymod & Mod::Control == Mod::Control),

                        ..Default::default()
                    };

                    if self.modifiers.command && key == egui::Key::X {
                        self.input.events.push(egui::Event::Cut);
                    } else if self.modifiers.command && key == egui::Key::C {
                        self.input.events.push(egui::Event::Copy);
                    } else if self.modifiers.command && key == egui::Key::V {
                        if let Some(clipboard_ctx) = self.clipboard.as_mut() {
                            self.input.events.push(egui::Event::Text(
                                clipboard_ctx
                                    .get_contents()
                                    .unwrap_or_else(|_| "".to_string()),
                            ));
                        }
                    } else {
                        self.input.events.push(Event::Key {
                            key,
                            pressed: true,
                            repeat: false,
                            modifiers: self.modifiers,
                        });
                    }
                }
            }

            Char(c) => {
                self.input.events.push(Event::Text(c.to_string()));
            }

            Scroll(x, y) => {
                self
                    .input
                    .events
                    .push(Event::Scroll(vec2(x as f32, y as f32)));
            }

            _ => {}
        }
    }

    pub fn handle_clipboard(&mut self, platform_output: PlatformOutput) {
        if !platform_output.copied_text.is_empty() {
            self.copy_to_clipboard(platform_output.copied_text);
        }
    }

    pub fn update(&mut self, glfw_window: &glfw::Window, elapsed_time_as_secs: f64) {
        let (width, height) = glfw_window.get_framebuffer_size();
        let native_pixels_per_point = glfw_window.get_content_scale().0;
        self.input.screen_rect = Some(egui::Rect::from_min_size(
            Pos2::new(0f32, 0f32),
            vec2(width as f32, height as f32) / native_pixels_per_point,
        ));

        self.input.time = Some(elapsed_time_as_secs);
        self.input.pixels_per_point = Some(native_pixels_per_point);
    }

    pub fn copy_to_clipboard(&mut self, copy_text: String) {
        if let Some(clipboard) = self.clipboard.as_mut() {
            let result = clipboard.set_contents(copy_text);
            if result.is_err() {
                dbg!("Unable to set clipboard content.");
            }
        }
    }

    pub fn take_raw_input(&mut self) -> RawInput {
        self.input.take()
    }

    fn init_clipboard() -> Option<ClipboardContext> {
        match ClipboardContext::new() {
            Ok(clipboard) => Some(clipboard),
            Err(err) => {
                eprintln!("Failed to initialize clipboard: {}", err);
                None
            }
        }
    }

    fn translate_glfwkey_to_eguikey(key: glfw::Key) -> Option<egui::Key> {
        use glfw::Key::*;

        Some(match key {
            Left => Key::ArrowLeft,
            Up => Key::ArrowUp,
            Right => Key::ArrowRight,
            Down => Key::ArrowDown,

            Escape => Key::Escape,
            Tab => Key::Tab,
            Backspace => Key::Backspace,
            Space => Key::Space,

            Enter => Key::Enter,

            Insert => Key::Insert,
            Home => Key::Home,
            Delete => Key::Delete,
            End => Key::End,
            PageDown => Key::PageDown,
            PageUp => Key::PageUp,

            A => Key::A,
            B => Key::B,
            C => Key::C,
            D => Key::D,
            E => Key::E,
            F => Key::F,
            G => Key::G,
            H => Key::H,
            I => Key::I,
            J => Key::J,
            K => Key::K,
            L => Key::L,
            M => Key::M,
            N => Key::N,
            O => Key::O,
            P => Key::P,
            Q => Key::Q,
            R => Key::R,
            S => Key::S,
            T => Key::T,
            U => Key::U,
            V => Key::V,
            W => Key::W,
            X => Key::X,
            Y => Key::Y,
            Z => Key::Z,

            _ => {
                return None;
            }
        })
    }

    fn translate_eguicursor_to_glfwcursor(cursor_icon: egui::CursorIcon) -> glfw::StandardCursor {
        match cursor_icon {
            CursorIcon::Default => glfw::StandardCursor::Arrow,

            CursorIcon::PointingHand => glfw::StandardCursor::Hand,

            CursorIcon::ResizeHorizontal => glfw::StandardCursor::HResize,
            CursorIcon::ResizeVertical => glfw::StandardCursor::VResize,
            // TODO: GLFW doesnt have these specific resize cursors, so we'll just use the HResize and VResize ones instead
            CursorIcon::ResizeNeSw => glfw::StandardCursor::HResize,
            CursorIcon::ResizeNwSe => glfw::StandardCursor::VResize,

            CursorIcon::Text => glfw::StandardCursor::IBeam,
            CursorIcon::Crosshair => glfw::StandardCursor::Crosshair,

            CursorIcon::Grab | CursorIcon::Grabbing => glfw::StandardCursor::Hand,

            // TODO: Same for these
            CursorIcon::NotAllowed | CursorIcon::NoDrop => glfw::StandardCursor::Arrow,
            CursorIcon::Wait => glfw::StandardCursor::Arrow,
            _ => glfw::StandardCursor::Arrow,
        }
    }
}

