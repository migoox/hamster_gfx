//! # gl and glfw egui integration
//! This module provides utility that allows simple gl and glfw integration with the egui library.
//! ## Usage
//! ```rust
//! use glfw::Context;
//! let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
//! // ...
//! let (mut window, events) = glfw
//!  .create_window(
//!             800,
//!             600,
//!             "Window",
//!             glfw::WindowMode::Windowed,
//!         )
//!         .expect("Failed to create the GLFW window");
//! // ...
//! let mut egui_ctx = egui::Context::default();
//! // ...
//! // Initialize egui integration
//! let mut egui_io = hamster_gfx::egui_integration::EguiIOHandler::new(&window);
//! let mut egui_painter = hamster_gfx::egui_integration::EguiPainter::new(&window);
//! // ...
//! let mut clock = std::time::Instant::now();
//! while !window.should_close() {
//!     for (_, event) in glfw::flush_messages(&events) {
//!         if event == glfw::WindowEvent::Close {
//!             window.set_should_close(true);
//!         } else {
//!             match event {
//!                 // GLFW event handling
//!                 _ => (),
//!             };
//!
//!             // Move GLFW events as an input into EguiIOHandler
//!             egui_io.handle_event(event);
//!         }
//!
//!     // Begin egui frame
//!     egui_ctx.begin_frame(egui_io.take_raw_input());
//!
//!     // Update egui integration
//!     egui_io.update(&window, clock.elapsed().as_secs_f64());
//!     egui_painter.update(&window);
//!
//!     // Do egui stuff (e.g. display an egui window with a button)
//!     // ...
//!
//!     // End egui frame
//!     let egui::FullOutput {
//!             platform_output,
//!             repaint_after: _,
//!             textures_delta,
//!             shapes,
//!         } = egui_ctx.end_frame();
//!
//!     // Handle platform output
//!     egui_io.handle_platform_output(platform_output, &mut window);
//!
//!     // Render OpenGL stuff (e.g. draw a triangle)
//!     // ...
//!
//!     // Render egui
//!     egui_painter.paint(&egui_ctx.tessellate(shapes), &textures_delta);
//!
//!     window.swap_buffers();
//!     glfw.poll_events();
//! }
//! ```
use std::path::Path;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use cli_clipboard::{ClipboardContext, ClipboardProvider};
use egui::*;
use gl::types::{GLint, GLsizei, GLvoid};
use glfw::Cursor;
use image::EncodableLayout;
use crate::gl_wrapper::{Shader, ShaderProgram, Buffer, VertexArray, Bindable, VertexBufferLayout, VertexAttrib, Texture, RenderTarget, RenderSettings};

struct TextureData {
    texture: Texture,
    srgb: bool,
}

impl TextureData {
    pub fn new(texture: Texture, srgb: bool) -> TextureData {
        TextureData {
            texture,
            srgb,
        }
    }
}

/// Allows egui rendering.
/// Egui Painter uses 0th texture unit. If you are going to use 0th
/// texture unit, remember to bind it once per draw call.
pub struct EguiPainter {
    shader_program: ShaderProgram,

    vao: VertexArray,
    vbo_pos: Buffer,
    vbo_col: Buffer,
    vbo_tex: Buffer,
    ebo: Buffer,

    render_target: RenderTarget,

    textures: Rc<RefCell<HashMap<TextureId, TextureData>>>,

    canvas_width: i32,
    canvas_height: i32,
    native_pixels_per_point: f32,
}

impl EguiPainter {
    /// Creates a new instance of EguiPainter.
    pub fn new(glfw_window: &glfw::Window) -> EguiPainter {
        let vs = Shader::compile_from_path(&Path::new("resources/egui_shaders/vertex.vert"), gl::VERTEX_SHADER)
            .expect("Painter couldn't load the vertex egui_shader");
        let fs = Shader::compile_from_path(&Path::new("resources/egui_shaders/fragment.frag"), gl::FRAGMENT_SHADER)
            .expect("Painter couldn't load the fragment egui_shader");

        let shader_program = ShaderProgram::link(&vs, &fs);

        let mut vao = VertexArray::new();
        let mut vbo_pos = Buffer::new(gl::ARRAY_BUFFER, gl::STREAM_DRAW);
        let mut vbo_col = Buffer::new(gl::ARRAY_BUFFER, gl::STREAM_DRAW);
        let mut vbo_tex = Buffer::new(gl::ARRAY_BUFFER, gl::STREAM_DRAW);

        let mut vbl = VertexBufferLayout::new();
        vbl.push_attrib(0, VertexAttrib::new(2, gl::FLOAT, gl::FALSE));
        vao.attach_vbo(&vbo_pos, &vbl, 0).unwrap();

        let mut vbl = VertexBufferLayout::new();
        vbl.push_attrib(1, VertexAttrib::new(4, gl::UNSIGNED_BYTE, gl::FALSE));
        vao.attach_vbo(&vbo_col, &vbl, 0).unwrap();

        let mut vbl = VertexBufferLayout::new();
        vbl.push_attrib(2, VertexAttrib::new(2, gl::FLOAT, gl::FALSE));
        vao.attach_vbo(&vbo_tex, &vbl, 0).unwrap();

        let mut result = EguiPainter {
            shader_program,
            vao,
            vbo_pos,
            vbo_col,
            vbo_tex,
            ebo: Buffer::new(gl::ELEMENT_ARRAY_BUFFER, gl::STREAM_DRAW),
            textures: Rc::new(RefCell::new(HashMap::new())),
            render_target: RenderTarget::build(RenderSettings {
                scissor_test: true,
                blend: true,
                blend_func_source_factor: gl::ONE,
                blend_func_destination_factor: gl::ONE_MINUS_SRC_ALPHA,
                ..Default::default()
            }),
            canvas_width: 0,
            canvas_height: 0,
            native_pixels_per_point: 1.0,
        };
        result.update(&glfw_window);
        result
    }

    /// Updates screen rectangle and native pixels per point.
    pub fn update(&mut self, glfw_window: &glfw::Window) {
        (self.canvas_width, self.canvas_height) = glfw_window.get_framebuffer_size();
        self.native_pixels_per_point = glfw_window.get_content_scale().0;
    }

    pub fn set_egui_scale(&self, ctx: &Context, scale: f32) {
        let mut style = (*ctx.style()).clone();
        style.text_styles = [
            (
                TextStyle::Heading,
                FontId::new(scale * 30.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Body,
                FontId::new(scale * 22.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Monospace,
                FontId::new(scale * 18.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Button,
                FontId::new(scale * 18.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Small,
                FontId::new(scale * 14.0, FontFamily::Proportional),
            ),
        ]
            .into();
        ctx.set_style(style);
    }

    /// Paints and updates egui textures.
    pub fn paint(&mut self, clipped_primitives: &Vec<ClippedPrimitive>, texture_delta: &TexturesDelta) {
        // Update textures if they are already uploaded to the OpenGL or upload them
        self.update_textures(texture_delta);

        // Prepare OpenGL
        self.render_target.set_viewport(0,0, self.canvas_width, self.canvas_height);
        self.render_target.set_scissor(0,0, self.canvas_width, self.canvas_height);

        // Prepare Uniforms
        self.shader_program.set_uniform2f(
            "u_screen_size",
            self.canvas_width as f32 / self.native_pixels_per_point,
            self.canvas_height as f32 / self.native_pixels_per_point,
        );

        self.shader_program.activate_sampler("u_sampler", 0).unwrap();

        // Iterate through the clipped primitives produced by egui and paint them
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
    }

    // Handles egui textures delta, uploads their content into OpenGL buffers or updates
    // sub-buffers.
    fn update_textures(&mut self, textures_delta: &TexturesDelta) {
        for (tex_id, image_delta) in &textures_delta.set {

            // Gather data that will be uploaded to OpenGL
            let data: Vec<u8> = match &image_delta.image {
                ImageData::Color(color_data) => {
                    assert_eq!(
                        color_data.width() * color_data.height(),
                        color_data.pixels.len(),
                        "Mismatch between texture size and texel count"
                    );

                    color_data.pixels.iter()
                        .flat_map(|c| c.to_array())
                        .collect()
                }
                ImageData::Font(font_data) => {
                    assert_eq!(
                        font_data.width() * font_data.height(),
                        font_data.pixels.len(),
                        "Mismatch between texture size and texel count"
                    );

                    // Documentation says that gamma set to 1.0 looks the best
                    let gamma = 1.0;
                    font_data.srgba_pixels(Some(gamma))
                        .flat_map(|c| c.to_array())
                        .collect()
                }
            };

            // Get width and height of the patch (or the whole new texture)
            let [width, height] = image_delta.image.size();

            if let Some([x_offset, y_offset]) = image_delta.pos {
                // If image_delta.pos is Some, this describes a patch of the whole image starting
                // at image_delta.pos so in this case we won't allocate anything new, and no entry
                // will be added to textures hash map.

                match self.textures.borrow_mut().get(&tex_id) {
                    Some(texture_data) => {
                        // Update a sub-region of an already allocated texture with the patch
                        texture_data.texture.tex_sub_image2d(
                            x_offset as GLint,
                            y_offset as GLint,
                            width as GLsizei,
                            height as GLsizei,
                            gl::RGBA,
                            &data,
                        );
                    }
                    // The texture should exist at this point
                    None => panic!("Failed to find egui texture {:?}", tex_id),
                };
            } else {
                // If image_delta.pos is None, this describes the whole new texture. In this
                // we allocate new texture.

                // Create a new OpenGL texture
                let mut texture = Texture::new(
                    gl::TEXTURE_2D,
                    gl::LINEAR,
                    gl::CLAMP_TO_EDGE,
                );

                // Buffer the data
                texture.tex_image2d(
                    width as GLsizei,
                    height as GLsizei,
                    gl::RGBA as GLint,
                    gl::RGBA,
                    &data,
                );

                // Insert egui texture, if there was a texture with the given tex_id
                // it will be dropped.
                self.textures.borrow_mut().insert(*tex_id, TextureData::new(texture, true));
            }
        }
    }

    fn paint_mesh(&mut self, mesh: &Mesh, clip_rect: &Rect) {
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

        let clip_min_x = clip_min_x.round() as i32;
        let clip_min_y = clip_min_y.round() as i32;
        let clip_max_x = clip_max_x.round() as i32;
        let clip_max_y = clip_max_y.round() as i32;
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

        // Prepare data
        let vertices_len = mesh.vertices.len();
        let mut positions: Vec<f32> = Vec::with_capacity(2 * vertices_len);
        let mut tex_coords: Vec<f32> = Vec::with_capacity(2 * vertices_len);
        let mut colors: Vec<u8> = Vec::with_capacity(4 * vertices_len);
        for v in &mesh.vertices {
            positions.push(v.pos.x);
            positions.push(v.pos.y);

            tex_coords.push(v.uv.x);
            tex_coords.push(v.uv.y);

            colors.push(v.color[0]);
            colors.push(v.color[1]);
            colors.push(v.color[2]);
            colors.push(v.color[3]);
        }

        // Bind vao
        self.vao.bind();

        // Update buffers
        self.ebo.buffer_data::<u32>(
            mesh.indices.len(),
            mesh.indices.as_ptr(),
        );

        self.vbo_pos.buffer_data::<f32>(
            positions.len(),
            positions.as_ptr(),
        );

        self.vbo_col.buffer_data::<u8>(
            colors.len(),
            colors.as_ptr(),
        );

        self.vbo_tex.buffer_data::<f32>(
            tex_coords.len(),
            tex_coords.as_ptr(),
        );

        // Bind texture associated with the mesh
        self.shader_program.bind();

        let mut srgb = false;
        match self.textures.borrow().get(&mesh.texture_id) {
            Some(texture_data) => {
                texture_data.texture.activate(0);
                srgb = texture_data.srgb;
            }
            // The texture should exist at this point
            None => panic!("Failed to find egui texture {:?}", mesh.texture_id),
        };
        self.vao.bind();
        self.vao.use_vbo(&self.vbo_pos);
        self.vao.use_vbo(&self.vbo_col);
        self.vao.use_vbo(&self.vbo_tex);
        self.vao.use_ebo(&self.ebo);

        self.render_target.draw_elements_with_settings(
            gl::TRIANGLES,
            mesh.indices.len(),
            &self.shader_program,
            RenderSettings {
                framebuffer_srgb: srgb,
                ..Default::default()
            }
        )
    }
}

/// Allows creating egui textures and passing them to [`egui::Image::new()`].
/// [`EguiUserTexture`] **cannot outlive** [`EguiPainter`] instance.
/// ## SRGB format
/// Egui uses textures stored in SRGB format, so everytime an egui texture
/// is in use, [`EguiPainter`] sets [`gl::FRAMEBUFFER_SRGB`] flag before the draw call.
/// For that reason [`EguiPainter`]
/// needs to know if your texture is in srgb format as well, if it is not `srgb` flag
/// must be set to `false`.
/// ## Example
/// ```rust
/// use hamster_gfx::egui_integration::{EguiUserTexture, EguiPainter, EguiIOHandler};
/// // ...
/// let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
/// // Create GLFW window
/// let (mut window, events) = glfw
///     .create_window(
///         800,
///         600,
///         "Window",
///         glfw::WindowMode::Windowed,
///     )
///     .expect("Failed to create the GLFW window");
/// // ...
/// let egui_ctx = egui::Context::default();
/// let mut egui_painter = EguiPainter::new(&window);
/// let mut egui_io = EguiIOHandler::new(&window);
/// // ...
/// // Load the texture
/// let mut egui_txt = EguiUserTexture::from_image_path(
///     &mut egui_painter,
///     &std::path::Path::new("path/to/your/image.png"),
///     false,
/// );
/// // ...
/// while !window.should_close() {
///     // ...
///     egui_ctx.begin_frame(egui_io.take_raw_input());
///
///     egui::Window::new("Egui window").show(&egui_ctx, |ui| {
///         // Display Image with the texture
///         ui.add(egui::Image::new(egui_txt.get_id(), egui_txt.get_size()));
///     });
///
///     let egui::FullOutput {
///         platform_output,
///         repaint_after: _,
///         textures_delta,
///         shapes,
///     } = egui_ctx.end_frame();
///     // ...
/// }
/// ```
///
pub struct EguiUserTexture {
    egui_txt_id: TextureId,
    textures: Rc<RefCell<HashMap<TextureId, TextureData>>>,
    width: usize,
    height: usize,
}

impl EguiUserTexture {
    /// Creates a new OpenGL texture and passes `data` to it's buffer. Returns
    /// new instance of an EguiUserTexture.
    pub fn new(
        painter: &mut EguiPainter,
        filtering: TextureFilter,
        width: usize,
        height: usize,
        srgb: bool,
        data: &[Color32],
    ) -> EguiUserTexture {
        assert_eq!(
            width * height,
            data.len(),
            "Mismatch between texture size and texel count"
        );

        let mut gl_texture = Texture::new(
            gl::TEXTURE_2D,
            match filtering {
                TextureFilter::Linear => gl::LINEAR,
                TextureFilter::Nearest => gl::NEAREST,
            },
            gl::CLAMP_TO_EDGE,
        );

        let data: Vec<u8> = data.iter().flat_map(|c| c.to_array()).collect();

        gl_texture.tex_image2d(
            width as GLint,
            height as GLint,
            gl::RGBA as GLint,
            gl::RGBA,
            data.as_bytes(),
        );

        let id = TextureId::User(painter.textures.borrow().len() as u64);
        painter.textures.borrow_mut().insert(id, TextureData::new(gl_texture, srgb));

        EguiUserTexture {
            egui_txt_id: id,
            textures: Rc::clone(&painter.textures),
            width,
            height,
        }
    }

    /// Takes ownership of the `gl_texture` and creates a new instance of an EguiUserTexture.
    pub fn from_gl_texture(painter: &mut EguiPainter, gl_texture: Texture, srgb: bool) -> Result<EguiUserTexture, String> {
        let (width, height) = match gl_texture.get_size() {
            Some(s) => (s.0, s.1),
            None => return Err("OpenGL Texture has no data.".to_string()),
        };

        let id = TextureId::User(painter.textures.borrow().len() as u64);
        painter.textures.borrow_mut().insert(id, TextureData::new(gl_texture, srgb));

        Ok(EguiUserTexture {
            egui_txt_id: id,
            textures: Rc::clone(&painter.textures),
            width,
            height,
        })
    }

    /// Creates a new OpenGL texture and passes 2D image specified by the `path` to it's buffer.
    /// Returns a new instance of an EguiUserTexture.
    pub fn from_image_path(painter: &mut EguiPainter, path: &Path, srgb: bool) -> Result<EguiUserTexture, String> {
        let mut gl_texture = Texture::new(gl::TEXTURE_2D, gl::LINEAR, gl::CLAMP_TO_EDGE);
        gl_texture.tex_image2d_from_path_no_flip(&Path::new("resources/images/hamster2.png")).unwrap();
        Self::from_gl_texture(painter, gl_texture, srgb)
    }

    /// Returns copy of the egui texture id.
    pub fn get_id(&self) -> TextureId {
        self.egui_txt_id
    }

    /// Returns texture size as [`egui::Vec2`].
    pub fn get_size(&self) -> Vec2 {
        vec2(self.width as f32, self.height as f32)
    }

    /// Updates OpenGL buffer (calls [`gl::TexSubImage2D()`]), with the given data.
    pub fn update(&self, data: &[Color32]) {
        assert_eq!(
            self.width * self.height,
            data.len(),
            "Mismatch between texture size and texel count"
        );

        let data: Vec<u8> = data.iter().flat_map(|c| c.to_array()).collect();

        match self.textures.borrow().get(&self.egui_txt_id) {
            Some(texture_data) => {
                texture_data.texture.tex_sub_image2d(
                    0,
                    0,
                    self.width as GLsizei,
                    self.height as GLsizei,
                    gl::RGBA,
                    &data,
                );
            }
            None => panic!("Egui texture is invalid"),
        };
    }
}

impl Drop for EguiUserTexture {
    fn drop(&mut self) {
        self.textures.borrow_mut().remove(&self.egui_txt_id);
    }
}

struct EguiCursorManager {
    last_cursor: Option<glfw::StandardCursor>,
}

impl EguiCursorManager {
    fn new() -> EguiCursorManager {
        EguiCursorManager {
            last_cursor: None,
        }
    }

    // Changes the current cursor
    pub fn set_cursor(&mut self, cursor_icon: egui::CursorIcon, glfw_window: &mut glfw::Window) {
        let st_cursor = Self::translate_eguicursor_to_glfwcursor(cursor_icon);

        if let Some(c) = self.last_cursor {
            if c == st_cursor {
                return;
            }
        }

        glfw_window.set_cursor(Some(Cursor::standard(st_cursor)));
        self.last_cursor = Some(st_cursor);
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

/// This structure allows translation of glfw events into egui events (input handling)
/// and egui platform output handling (output handling).
pub struct EguiIOHandler {
    pointer_pos: Pos2,
    clipboard: Option<ClipboardContext>,
    input: RawInput,
    modifiers: Modifiers,
    cursor_manager: EguiCursorManager,
}

impl EguiIOHandler {
    /// Creates a new instance of an EguiIOHandler.
    pub fn new(glfw_window: &glfw::Window) -> EguiIOHandler {
        let (width, height) = glfw_window.get_framebuffer_size();
        EguiIOHandler {
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
            cursor_manager: EguiCursorManager::new(),
        }
    }

    /// Translates glfw events into egui events and pushes them onto egui event queue.
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

    /// Handles egui platform output, that is returned every frame by [`egui::Context::end_frame()`] and [`egui::Context::run()`] functions.
    /// Mutable window borrow is required, since this function may change cursor image.
    pub fn handle_platform_output(&mut self, platform_output: PlatformOutput, glfw_window: &mut glfw::Window) {
        if !platform_output.copied_text.is_empty() {
            self.copy_to_clipboard(platform_output.copied_text);
        }
        self.cursor_manager.set_cursor(platform_output.cursor_icon, glfw_window);
    }

    /// Updates screen rectangle, native pixels per point and elapsed time.
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

    /// Returns [`egui::RawInput`] that should be moved into [`egui::Context::run()`] or
    /// [`egui::Context::begin_frame()`] function.
    pub fn take_raw_input(&mut self) -> RawInput {
        self.input.take()
    }

    // Creates cli-clipboard context.
    fn init_clipboard() -> Option<ClipboardContext> {
        match ClipboardContext::new() {
            Ok(clipboard) => Some(clipboard),
            Err(err) => {
                eprintln!("Failed to initialize clipboard: {}", err);
                None
            }
        }
    }

    // Uses cli-clipboard library in order to put copied text into the system clipboard.
    fn copy_to_clipboard(&mut self, copy_text: String) {
        if let Some(clipboard) = self.clipboard.as_mut() {
            let result = clipboard.set_contents(copy_text);
            if result.is_err() {
                dbg!("Unable to set clipboard content.");
            }
        }
    }

    // Translates glfw key codes into egui key codes.
    fn translate_glfwkey_to_eguikey(key: glfw::Key) -> Option<egui::Key> {
        // From: https://github.com/cohaereo/egui_glfw_gl
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
}

