use std::sync::{OnceLock, Mutex};
use std::io;
use gl::types::{GLintptr, GLboolean, GLbyte, GLchar, GLenum, GLfloat, GLint, GLsizei, GLsizeiptr, GLuint};
use std::ffi::{c_void, CString};
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;
use std::path::Path;
use image::ImageError;
use crate::errors::check_opengl_errors;

struct Binder {
    curr_id: GLuint,
}

impl Binder {
    pub fn new() -> Binder {
        Binder {
            curr_id: 0,
        }
    }

    pub fn bind<F>(&mut self, id: GLuint, binding_closure: F) -> bool
        where
            F: FnOnce(GLuint)
    {
        if id == self.curr_id {
            return false;
        }

        self.curr_id = id;
        binding_closure(id);
        true
    }

    pub fn unbind<F>(&mut self, unbinding_closure: F) -> bool
        where
            F: FnOnce()
    {
        if self.curr_id == 0 {
            return false;
        }

        unbinding_closure();
        true
    }

    pub fn get_bound_id(&self) -> GLuint {
        self.curr_id
    }
}

struct MultipleTargetBinder {
    target_id_map: HashMap<GLenum, GLuint>,
}

impl MultipleTargetBinder {
    pub fn new() -> MultipleTargetBinder {
        MultipleTargetBinder {
            target_id_map: HashMap::new(),
        }
    }

    pub fn bind<F>(&mut self, target: GLenum, id: GLuint, binding_closure: F) -> bool
        where
            F: FnOnce(GLenum, GLuint)
    {
        let mut prev: GLuint = 0;
        if let Some(p) = self.target_id_map.get(&target) {
            prev = *p;
        }

        if id == prev {
            return false;
        }

        self.target_id_map.entry(target)
            .and_modify(|x| { *x = id })
            .or_insert(id);

        binding_closure(target, id);
        true
    }

    pub fn unbind<F>(&mut self, target: GLenum, unbinding_closure: F) -> bool
        where
            F: FnOnce(GLenum)
    {
        let prev = match self.target_id_map.get(&target) {
            Some(p) => *p,
            None => 0,
        };

        if prev == 0 {
            return false;
        }

        self.target_id_map.entry(target).and_modify(|i| { *i = 0 });
        unbinding_closure(target);
        true
    }

    pub fn get_bound_id(&mut self, target: GLenum) -> GLuint {
        match self.target_id_map.get(&target) {
            Some(id) => *id,
            None => 0,
        }
    }
}

pub trait Bindable {
    /// This function binds an OpenGL object if it's not already bound. If the object is
    /// already bound, it will return false without calling the OpenGL bind function. If the
    /// OpenGL bind function is called, bind() will return true.
    fn bind(&self) -> bool;
    /// Works like bind, but uses `0` as an id.
    fn unbind(&self) -> bool;
}

/// Shader
pub struct Shader {
    id: GLuint,
    stype: GLenum,
}

impl Shader {
    /// Compiles egui_shaders's source code contained in a file, accessed by the given `path`.
    /// The compiled Shader has the type specified by `shader_type`.
    /// Function will panic with the OpenGL egui_shaders info log if compilation fails.
    pub fn compile_from_path(path: &Path, shader_type: GLenum) -> io::Result<Shader> {
        let mut file = File::open(path)?;
        let mut src = String::new();
        file.read_to_string(&mut src)?;

        Ok(Self::compile(&src, shader_type))
    }

    /// Compiles egui_shaders's source code `src`. The compiled Shader has the type specified
    /// by `shader_type`. Function will panic with the OpenGL egui_shaders info log if
    /// compilation fails.
    pub fn compile(src: &str, shader_type: GLenum) -> Shader {
        let shader = unsafe { gl::CreateShader(shader_type) };

        let c_str = CString::new(src.as_bytes()).unwrap();
        unsafe {
            gl::ShaderSource(shader, 1, &c_str.as_ptr(), core::ptr::null());
            gl::CompileShader(shader);
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        let mut status = gl::FALSE as GLint;
        unsafe {
            gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);
        }

        if status != (gl::TRUE as GLint) {
            let mut len = 0;
            unsafe {
                gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            }

            let mut buf = vec![0; len as usize];

            unsafe {
                gl::GetShaderInfoLog(
                    shader,
                    len,
                    core::ptr::null_mut(),
                    buf.as_mut_ptr() as *mut GLchar,
                );
            }

            panic!(
                "{}",
                core::str::from_utf8(&buf).expect("ShaderInfoLog not valid utf8")
            );
        }

        Shader {
            id: shader,
            stype: shader_type,
        }
    }

    /// Returns egui_shaders type (fragment/vertex).
    fn get_type(&self) -> GLenum {
        self.stype
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { gl::DeleteShader(self.id) }
    }
}

/// Shader Program
pub struct ShaderProgram {
    id: GLuint,
}

impl ShaderProgram {
    /// Combines `vert_shader` and `frag_shader` in order to create an OpenGl egui_shaders program object
    /// and binds the created ShaderProgram.
    /// Function will panic with an OpenGL program info log in the case of failure.
    /// Both vert_shader and frag_shader can be deleted after this call.
    pub fn link(vert_shader: &Shader, frag_shader: &Shader) -> ShaderProgram {
        assert!(
            vert_shader.get_type() == gl::VERTEX_SHADER && frag_shader.get_type() == gl::FRAGMENT_SHADER,
            "ShaderProgram link error -- vertex egui_shaders and fragment egui_shaders should be provided (types doesn't match)"
        );

        let program = unsafe { gl::CreateProgram() };

        unsafe {
            gl::AttachShader(program, vert_shader.id);
            gl::AttachShader(program, frag_shader.id);
            gl::LinkProgram(program);
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        let mut status = gl::FALSE as GLint;
        unsafe {
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        // If program cannot be linked -> panic
        if status != (gl::TRUE as GLint) {
            let mut len: GLint = 0;
            unsafe {
                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            }

            let mut buf = vec![0; len as usize];

            unsafe {
                gl::GetProgramInfoLog(
                    program,
                    len,
                    core::ptr::null_mut(),
                    buf.as_mut_ptr() as *mut GLchar,
                );
            }

            panic!(
                "{}",
                core::str::from_utf8(&buf).expect("ProgramInfoLog not valid utf8")
            );
        }

        let sprogram = ShaderProgram {
            id: program,
        };
        sprogram.bind();
        sprogram
    }

    /// Sets glsl's sampler uniform with `name` to `unit`.
    /// This function calls set_uniform1i, which uses gl::ProgramUniform* command family, which means
    /// that there is no need to bind ShaderProgram, so this function won't bind anything.
    pub fn activate_sampler(&self, name: &str, unit: u32) -> Result<(), String> {
        self.set_uniform1i(name, unit as i32);
        Ok(())
    }

    /// This function is a gl::ProgramUniform* wrapper.
    pub fn set_uniform1ui(&self, name: &str, v0: u32) {
        let cname = CString::new(name.as_bytes()).unwrap();
        unsafe {
                gl::ProgramUniform1ui(
                self.id,
                gl::GetUniformLocation(self.id, cname.as_ptr()),
                v0,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    /// This function is a gl::ProgramUniform* wrapper.
    pub fn set_uniform2ui(&self, name: &str, v0: u32, v1: u32) {
        let cname = CString::new(name.as_bytes()).unwrap();
        unsafe {
            gl::ProgramUniform2ui(
                self.id,
                gl::GetUniformLocation(self.id, cname.as_ptr()),
                v0,
                v1,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    /// This function is a gl::ProgramUniform* wrapper.
    pub fn set_uniform1i(&self, name: &str, v0: i32) {
        let cname = CString::new(name.as_bytes()).unwrap();
        unsafe {
            gl::ProgramUniform1i(
                self.id,
                gl::GetUniformLocation(self.id, cname.as_ptr()),
                v0,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    /// This function is a gl::ProgramUniform* wrapper.
    pub fn set_uniform2i(&self, name: &str, v0: i32, v1: i32) {
        let cname = CString::new(name.as_bytes()).unwrap();
        unsafe {
            gl::ProgramUniform2i(
                self.id,
                gl::GetUniformLocation(self.id, cname.as_ptr()),
                v0,
                v1,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    /// This function is a gl::ProgramUniform* wrapper.
    pub fn set_uniform1f(&self, name: &str, v0: f32) {
        let cname = CString::new(name.as_bytes()).unwrap();
        unsafe {
            gl::ProgramUniform1f(
                self.id,
                gl::GetUniformLocation(self.id, cname.as_ptr()),
                v0,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    /// This function is a gl::ProgramUniform* wrapper.
    pub fn set_uniform2f(&self, name: &str, v0: f32, v1: f32) {
        let cname = CString::new(name.as_bytes()).unwrap();
        unsafe {
            gl::ProgramUniform2f(
                self.id,
                gl::GetUniformLocation(self.id, cname.as_ptr()),
                v0,
                v1,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    /// This function is a gl::ProgramUniform* wrapper.
    pub fn set_uniform3f(&self, name: &str, v0: f32, v1: f32, v2: f32) {
        let cname = CString::new(name.as_bytes()).unwrap();
        unsafe {
            gl::ProgramUniform3f(
                self.id,
                gl::GetUniformLocation(self.id, cname.as_ptr()),
                v0,
                v1,
                v2,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    /// This function is a gl::ProgramUniform* wrapper.
    pub fn set_uniform4f(&self, name: &str, v0: f32, v1: f32, v2: f32, v3: f32) {
        let cname = CString::new(name.as_bytes()).unwrap();
        unsafe {
            gl::ProgramUniform4f(
                self.id,
                gl::GetUniformLocation(self.id, cname.as_ptr()),
                v0,
                v1,
                v2,
                v3,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    // Binder shared between instances of the ShaderProgram structure
    fn binder() -> &'static Mutex<Binder> {
        static BINDER: OnceLock<Mutex<Binder>> = OnceLock::new();
        BINDER.get_or_init(|| Mutex::new(Binder::new()))
    }
}

impl Bindable for ShaderProgram {
    fn bind(&self) -> bool {
        Self::binder().lock().unwrap().bind(self.id, |i| {
            unsafe {
                gl::UseProgram(i);
            }
        })
    }

    fn unbind(&self) -> bool {
        Self::binder().lock().unwrap().unbind(|| {
            unsafe {
                gl::UseProgram(0);
            }
        })
    }
}

/// Texture
pub struct Texture {
    id: GLuint,
    // It is not legal to bind an object to a different target than the
    // one it was previously bound with.
    target: GLenum,
    filtering: GLenum,
    wrapping: GLenum,
}

impl Texture {
    /// Creates and binds to the `target` a new texture.
    pub fn new(
        target: GLenum,
        filtering: GLenum,
        wrapping: GLenum,
    ) -> Texture {
        let mut id: GLuint = 0;
        unsafe {
            gl::GenTextures(0, &mut id);
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        let result = Texture {
            id,
            target,
            filtering,
            wrapping,
        };

        result.bind();

        unsafe {
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, result.wrapping as GLint);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, result.wrapping as GLint);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, result.filtering as GLint);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, result.filtering as GLint);
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        result
    }

    /// Calls gl::TexImage2D.
    pub fn tex_image2d(
        &self,
        width: GLint,
        height: GLint,
        format: GLuint,
        pixels: *const c_void,
    ) -> Result<(), String> {
        self.bind();

        unsafe {
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGB as GLint,
                width,
                height,
                0,
                format,
                gl::UNSIGNED_BYTE,
                pixels,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        Ok(())
    }

    /// Load image from the given path (uses image crate) and supply the gl::TexImage2D
    /// with it.
    pub fn tex_image2d_from_path(&self, path: &Path) -> Result<(), ImageError> {
        self.bind();

        let img = image::open(path)?;

        // Flip the image vertically
        let img = image::imageops::flip_vertical(&img);

        unsafe {
            use image::EncodableLayout;
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGBA as i32,
                img.width() as i32,
                img.height() as i32,
                0,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                img.as_bytes().as_ptr() as *const _,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        Ok(())
    }

    /// Activates the texture on the given texture `unit`.
    pub fn activate(&self, unit: u32) {
        assert!(
            unit < gl::MAX_COMBINED_TEXTURE_IMAGE_UNITS as u32,
            "{}",
            format!("Texture slot must be >= 0 and < {}", gl::MAX_COMBINED_TEXTURE_IMAGE_UNITS as u32),
        );
        // From: https://www.khronos.org/opengl/wiki/Texture
        //
        // ,,Binding textures for use in OpenGL is a little weird. There are two reasons
        // to bind a texture object to the context: to change the object (e.g. modify
        // its storage or its parameters) or to render something with it.''
        //

        unsafe {
            gl::ActiveTexture(((gl::TEXTURE0 as u32) + unit) as GLenum);

            // Bind texture globally (bind it to the context in order to modify storage/parameters)
            // This call is necessary, since binder should keep track of the current global bind
            let gl_bind_called = self.bind();

            // Bind texture to the unit in order to render something with it
            // This call is necessary, since self.bind() doesn't have to call gl::BindTexture
            if gl_bind_called == false {
                gl::BindTexture(self.target, self.id);
            }
        }
    }

    // Define binder shared between instances of the Texture structure
    fn binder() -> &'static Mutex<MultipleTargetBinder> {
        static BINDER: OnceLock<Mutex<MultipleTargetBinder>> = OnceLock::new();
        BINDER.get_or_init(|| { Mutex::new(MultipleTargetBinder::new()) })
    }
}

impl Bindable for Texture {
    fn bind(&self) -> bool {
        Self::binder().lock().unwrap().bind(
            self.target,
            self.id,
            |t, i| {
                unsafe {
                    gl::BindTexture(t, i);
                }
            })
    }

    fn unbind(&self) -> bool {
        Self::binder().lock().unwrap().unbind(
            self.target,
            |t| {
                unsafe {
                    gl::BindTexture(t, 0);
                }
            })
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteTextures(1, &self.id);
        }
    }
}

/// VertexAttrib represents one attribute of a vertex in the vertex buffer.
pub struct VertexAttrib {
    pub size: GLint,
    pub type_: GLenum,
    pub normalized: GLboolean,
}

impl VertexAttrib {
    /// Creates a VertexAttrib with the given parameters.
    pub fn new(size: GLint, type_: GLenum, normalized: GLboolean) -> VertexAttrib {
        VertexAttrib {
            size,
            type_,
            normalized,
        }
    }

    /// Returns real bytes size of the attribute.
    pub fn get_size_in_bytes(&self) -> usize {
        (self.size as usize) * match self.type_ {
            gl::FLOAT => std::mem::size_of::<GLfloat>(),
            gl::UNSIGNED_INT => std::mem::size_of::<GLuint>(),
            gl::UNSIGNED_BYTE => std::mem::size_of::<GLbyte>(),
            _ => panic!("Unsupported type"),
        }
    }
}

/// BufferLayout represents all attributes of a vertex in the vertex buffer.
pub struct VertexBufferLayout {
    attribs: Vec<(GLuint, VertexAttrib)>,
    stride: GLsizei,
}

impl VertexBufferLayout {
    /// Creates a new instance of the VertexBufferLayout.
    pub fn new() -> VertexBufferLayout {
        VertexBufferLayout {
            attribs: Vec::new(),
            stride: 0,
        }
    }

    /// Pushes (moves) attribute on the attrib stack (order matters) and updates stride.
    /// `attrib_index` specifies layout location used in GLSL's vertex egui_shaders.
    pub fn push_attrib(&mut self, attrib_index: GLuint, attrib: VertexAttrib) -> &mut VertexBufferLayout {
        self.stride += attrib.get_size_in_bytes() as GLsizei;
        self.attribs.push((attrib_index, attrib));
        self
    }
}

/// Buffer is a wrapper over the OpenGL Buffer Object. It can become vertex buffer/element buffer,
/// which depends on the given target.
pub struct Buffer {
    id: GLuint,
    target: GLenum,
    usage: GLenum,
}

impl Buffer {
    /// Generates the OpenGL Buffer Object.
    pub fn new(target: GLenum, usage: GLenum) -> Buffer {
        let mut id: GLuint = 0;

        unsafe {
            gl::GenBuffers(1, &mut id);
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        let result = Buffer {
            id,
            target,
            usage,
        };

        result.bind();
        result
    }

    /// Builds and binds OpenGL Buffer Object (uses gl::BufferData).
    pub fn build(target: GLenum, size: usize, usage: GLenum, data: *const c_void) -> Buffer {
        let mut result = Self::new(target, usage);

        result.bind();

        unsafe {
            gl::BufferData(target, size as GLsizeiptr, data, usage);
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        result
    }

    /// Builds empty buffer with the given `size` (uses gl::BufferData).
    pub fn build_empty(target: GLenum, usage: GLenum, size: usize) -> Buffer {
        Self::build(target, size, usage, core::ptr::null())
    }

    /// Calls gl::BufferData. If a different VBO is currently binded, this function
    /// will bind `self`. Previous binding will not be restored!
    pub fn buffer_data(&self, size: usize, data: *const c_void) -> Result<(), String> {
        self.bind();
        unsafe {
            gl::BufferData(self.target, size as _, data, self.usage);
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        Ok(())
    }

    /// Calls gl::BufferSubData. If a different VBO is currently binded, this function
    /// will bind `self` and do it's work. Previous binding will not be restored!
    pub fn buffer_sub_data(&self, size: usize, offset: u32, data: *const c_void) -> Result<(), String> {
        self.bind();
        unsafe {
            gl::BufferSubData(self.target, offset as _, size as _, data);
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        Ok(())
    }

    // Define binder shared between instances of Buffer object
    fn binder() -> &'static Mutex<MultipleTargetBinder> {
        static BINDER: OnceLock<Mutex<MultipleTargetBinder>> = OnceLock::new();
        BINDER.get_or_init(|| { Mutex::new(MultipleTargetBinder::new()) })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.id);
        }
    }
}

impl Bindable for Buffer {
    fn bind(&self) -> bool {
        Self::binder().lock().unwrap().bind(
            self.target,
            self.id,
            |t, i| {
                unsafe {
                    gl::BindBuffer(t, i);
                }
            })
    }

    fn unbind(&self) -> bool {
        Self::binder().lock().unwrap().unbind(
            self.target,
            |t| {
                unsafe {
                    gl::BindBuffer(t, 0);
                }
            })
    }
}

/// VertexArray is a wrapper over the OpenGL Vertex Array Object.
pub struct VertexArray {
    id: GLuint,
    // Stores vbo info (stride, offset, format binding index)
    vbo_info: HashMap<GLuint, (GLsizei, GLintptr, GLuint)>,
    curr_binding_index: GLuint,
}

impl VertexArray {
    /// Creates a new instance of the OpenGL Vertex Array Object.
    pub fn new() -> VertexArray {
        let mut id: GLuint = 0;

        unsafe { gl::GenVertexArrays(1, &mut id); }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        let result = VertexArray {
            id,
            vbo_info: HashMap::new(),
            curr_binding_index: 0,
        };

        result.bind();

        result
    }

    // https://computergraphics.stackexchange.com/questions/4637/can-one-vao-store-multiple-calls-to-glvertexattribpointer
    /// Binds the VAO (without restoring the previous binding) and associates the given VBO's id
    /// with the `layout` (OpenGL calls: gl::VertexAttribFormat, gl::VertexAttribBinding,
    /// gl::EnableVertexAttribArray). This function must be called exactly once before use_vbo calls.
    pub fn attach_vbo(&mut self, vbo: &Buffer, layout: &VertexBufferLayout, offset: GLintptr) -> Result<(), String> {
        self.bind();
        let mut relative_offset: GLuint = 0;
        for (attrib_index, attrib) in &layout.attribs {
            unsafe {
                gl::VertexAttribFormat(
                    *attrib_index,
                    attrib.size,
                    attrib.type_,
                    attrib.normalized,
                    relative_offset,
                );
                gl::VertexAttribBinding(*attrib_index, self.curr_binding_index);
                gl::EnableVertexAttribArray(*attrib_index);

                #[cfg(feature = "gl_debug")]
                check_opengl_errors();
            }
            relative_offset += attrib.get_size_in_bytes() as GLuint;
        }
        self.vbo_info.insert(vbo.id, (layout.stride, offset, self.curr_binding_index));
        self.curr_binding_index += 1;
        Ok(())
    }

    /// Binds the VAO (without restoring the previous binding) and uses gl::BindVertexBuffer
    /// with the layout associated with the VBO (see `attach_vbo`). This call should be
    /// used before rendering the VAO.
    pub fn use_vbo(&self, vbo: &Buffer) {
        self.bind();
        unsafe {
            gl::BindVertexBuffer(
                self.vbo_info.get(&vbo.id).unwrap().2,
                vbo.id,
                self.vbo_info.get(&vbo.id).unwrap().1,
                self.vbo_info.get(&vbo.id).unwrap().0,
            );
            #[cfg(feature = "gl_debug")]
            check_opengl_errors();
        }
    }

    /// Binds the VAO (without restoring the previous binding) and the given Element Buffer
    /// Object (`ebo`).
    pub fn use_ebo(&self, ebo: &Buffer) {
        self.bind();
        ebo.bind();
    }

    // Binder shared between instances of the VertexArray structure
    fn binder() -> &'static Mutex<Binder> {
        static BINDER: OnceLock<Mutex<Binder>> = OnceLock::new();
        BINDER.get_or_init(|| Mutex::new(Binder::new()))
    }
}

impl Bindable for VertexArray {
    fn bind(&self) -> bool {
        Self::binder().lock().unwrap().bind(self.id, |i| {
            unsafe {
                gl::BindVertexArray(i);
            }
        })
    }

    fn unbind(&self) -> bool {
        Self::binder().lock().unwrap().unbind(|| {
            unsafe {
                gl::BindVertexArray(0);
            }
        })
    }
}

impl Drop for VertexArray {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteVertexArrays(1, &self.id);
        }
    }
}