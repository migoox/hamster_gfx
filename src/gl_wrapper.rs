use std::sync::{OnceLock, Mutex};
use std::io;
use gl::types::{GLintptr, GLboolean, GLbyte, GLchar, GLenum, GLfloat, GLint, GLsizei, GLsizeiptr, GLuint, GLvoid};
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
        self.set_uniform_1i(name, unit as i32);
        Ok(())
    }

    /// This function is a gl::ProgramUniform* wrapper.
    pub fn set_uniform_1ui(&self, name: &str, v0: u32) {
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
    pub fn set_uniform_2ui(&self, name: &str, v0: u32, v1: u32) {
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
    pub fn set_uniform_1i(&self, name: &str, v0: i32) {
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
    pub fn set_uniform_2i(&self, name: &str, v0: i32, v1: i32) {
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
    pub fn set_uniform_1f(&self, name: &str, v0: f32) {
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
    pub fn set_uniform_2f(&self, name: &str, v0: f32, v1: f32) {
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
    pub fn set_uniform_3f(&self, name: &str, v0: f32, v1: f32, v2: f32) {
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
    pub fn set_uniform_4f(&self, name: &str, v0: f32, v1: f32, v2: f32, v3: f32) {
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
    pub fn set_uniform_mat4fv(&self, name: &str, mat: &glam::Mat4) {
        let cname = CString::new(name.as_bytes()).unwrap();
        unsafe {
            gl::ProgramUniformMatrix4fv(
                self.id,
                gl::GetUniformLocation(self.id, cname.as_ptr()),
                1,
                gl::FALSE,
                &mat.to_cols_array()[0],
            );
        }
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

    internal_format: Option<GLenum>,
    src_format: Option<GLenum>,
    type_: Option<GLenum>,

    width: Option<usize>,
    height: Option<usize>,
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
            gl::GenTextures(1, &mut id);
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();

        let result = Texture {
            id,
            target,
            filtering,
            wrapping,
            internal_format: None,
            src_format: None,
            type_: None,
            width: None,
            height: None,
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
        &mut self,
        width: usize,
        height: usize,
        internal_format: GLenum,
        src_format: GLenum,
        bytes: &[u8],
    ) {
        self.raw_tex_image2d(
            width,
            height,
            internal_format,
            src_format,
            gl::UNSIGNED_BYTE,
            bytes.as_ptr() as * const _,
        );
    }

    pub fn tex_image2d_null(
        &mut self,
        width: usize,
        height: usize,
        internal_format: GLenum,
        src_format: GLenum,
        type_: GLenum,
    ) {
        self.raw_tex_image2d(
            width,
            height,
            internal_format,
            src_format,
            type_,
            std::ptr::null() as *const c_void,
        );
    }

    pub fn raw_tex_image2d(
        &mut self,
        width: usize,
        height: usize,
        internal_format: GLenum,
        src_format: GLenum,
        type_: GLenum,
        ptr: *const c_void,
    ) {
        self.bind();

        self.width = Some(width);
        self.height = Some(height);

        self.internal_format = Some(internal_format);
        self.src_format = Some(src_format);
        self.type_ = Some(type_);

        unsafe {
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                internal_format as GLint,
                width as GLint,
                height as GLint,
                0,
                src_format,
                type_,
                ptr,
            );
        }

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    pub fn get_size(&self) -> Option<(usize, usize)> {
        Some((
            match self.width {
                Some(w) => w,
                None => return None,
            },
            match self.height {
                Some(h) => h,
                None => return None,
            }
        ))
    }

    pub fn get_internal_format(&self) -> Option<GLenum> {
        self.internal_format.clone()
    }

    pub fn get_src_format(&self) -> Option<GLenum> {
        self.src_format.clone()
    }

    pub fn get_type(&self) -> Option<GLenum> {
        self.type_.clone()
    }

    pub fn get_filtering(&self) -> GLenum {
        self.filtering
    }

    /// Load image from the given path (uses image crate) and supply the gl::TexImage2D
    /// with it.
    pub fn tex_image2d_from_path(&mut self, path: &Path) -> Result<(), ImageError> {
        self.bind();

        let img = image::open(path)?;

        // Flip the image vertically
        let img = image::imageops::flip_vertical(&img);

        self.width = Some(img.width() as usize);
        self.height = Some(img.height() as usize);

        self.internal_format = Some(gl::RGBA);
        self.src_format = Some(gl::RGBA);
        self.type_ = Some(gl::UNSIGNED_BYTE);

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
    pub fn tex_image2d_from_path_no_flip(&mut self, path: &Path) -> Result<(), ImageError> {
        self.bind();

        let img = image::open(path)?;

        let img = img.to_rgba8();

        self.width = Some(img.width() as usize);
        self.height = Some(img.height() as usize);

        self.internal_format = Some(gl::RGBA);
        self.src_format = Some(gl::RGBA);
        self.type_ = Some(gl::UNSIGNED_BYTE);

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
    pub fn tex_sub_image2d(
        &self,
        x_offset: usize,
        y_offset: usize,
        width: usize,
        height: usize,
        format: GLuint,
        bytes: &[u8],
    ) {
        self.bind();
        unsafe {
            gl::TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                x_offset as _,
                y_offset as _,
                width as _ ,
                height as _,
                format,
                self.type_.expect("Type is unknown."),
                bytes.as_ptr() as *const _,
            )
        }
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
            if !gl_bind_called {
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

    size: usize,
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
            size: 0,
        };

        result.bind();
        result
    }

    /// Builds and binds OpenGL Buffer Object (uses gl::BufferData).
    pub fn build<T>(target: GLenum, usage: GLenum, size: usize, data: *const T) -> Buffer {
        let mut result = Self::new(target, usage);
        result.buffer_data::<T>(size, data);

        result
    }

    /// Calls gl::BufferData. If a different VBO is currently binded, this function
    /// will bind `self`. Previous binding will not be restored!
    pub fn buffer_data<T>(&mut self, size: usize, data: *const T) {
        self.bind();
        unsafe {
            gl::BufferData(
                self.target,
                (size * std::mem::size_of::<T>()) as GLsizeiptr,
                data as *const c_void,
                self.usage
            );
        }
        self.size = size;

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    /// Calls gl::BufferSubData. If a different VBO is currently binded, this function
    /// will bind `self` and do it's work. Previous binding will not be restored!
    pub fn buffer_sub_data<T>(&mut self, size: usize, offset: usize, data: *const c_void) {
        self.bind();
        unsafe {
            gl::BufferSubData(
                self.target,
                (offset * std::mem::size_of::<T>()) as GLintptr,
                (size * std::mem::size_of::<T>()) as GLsizeiptr,
                data
            );
        }
        self.size = size;

        #[cfg(feature = "gl_debug")]
        check_opengl_errors();
    }

    pub fn get_size(&self) -> usize {
        self.size
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
        let info = self.vbo_info.get(&vbo.id)
            .expect("Vertex Array Object can't use not attached Vertex Buffer if it's not attached.");

        unsafe {
            gl::BindVertexBuffer(
                info.2,
                vbo.id,
                info.1,
                info.0,
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

pub struct FrameBuffer {
    id: GLuint,

    width: usize,
    height: usize,

    default: bool,

    depth_texture: Option<Texture>,
    color_texture: Option<Texture>,
}

impl FrameBuffer {
    /// Returns handle to the default framebuffer with id 0
    pub fn get_onscreen() -> FrameBuffer {
        FrameBuffer {
            id: 0,
            default: true,
            width: 0,
            height: 0,
            depth_texture: None,
            color_texture: None,
        }
    }

    /// Returns handle to the offscreen framebuffer with id > 0
    pub fn new_offscreen(width:usize, height: usize) -> FrameBuffer {
        let mut id = 0;
        unsafe {
            gl::GenFramebuffers(1, &mut id);
        }
        FrameBuffer {
            id,
            default: false,
            width,
            height,
            depth_texture: None,
            color_texture: None
        }
    }

    /// Binds and checks whether the framebuffer is ready to use. (Uses glCheckFrameBufferStatus)
    pub fn is_completed(&self) -> bool {
        self.bind();
        unsafe {
            return gl::CheckFramebufferStatus(gl::FRAMEBUFFER) == gl::FRAMEBUFFER_COMPLETE;
        }
    }

    pub fn attach_depth_buffer(&mut self) {
        assert!(!self.default, "Can't attach depth buffer to the default framebuffer");

        self.bind();

        let mut depth_texture = Texture::new(
            gl::TEXTURE_2D,
            gl::LINEAR,
            gl::CLAMP_TO_EDGE,
        );

        depth_texture.tex_image2d_null(
            self.width,
            self.height,
            gl::DEPTH_COMPONENT,
            gl::DEPTH_COMPONENT,
            gl::FLOAT
        );

        unsafe {
            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::DEPTH_ATTACHMENT,
                gl::TEXTURE_2D,
                depth_texture.id,
                0
            );
        }

        self.depth_texture = Some(depth_texture);
    }

    pub fn attach_color_buffer(
        &mut self,
        texture_filter: GLenum,
        texture_internal_format: GLenum,
        texture_src_format: GLenum,
        texture_type: GLenum,
    ) {
        assert!(!self.default, "Can't attach color buffer to the default framebuffer");

        self.bind();

        let mut color_texture = Texture::new(
            gl::TEXTURE_2D,
            texture_filter,
            gl::CLAMP_TO_EDGE,
        );

        color_texture.tex_image2d_null(
            self.width,
            self.height,
            texture_internal_format,
            texture_src_format,
            texture_type,
        );

        unsafe {
            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                color_texture.id,
                0
            );
        }

        self.color_texture = Some(color_texture);
    }

    /// Allows reading pixel from the color buffer.
    pub fn read_pixel(&self, x: usize, y: usize, ptr: *mut c_void) {
        assert!(0 <= x && x < self.width &&
                    0 <= y && y < self.height,
                "Pixel coordinates are out of bounds.");

        self.bind();

        unsafe {
            gl::ReadBuffer(gl::COLOR_ATTACHMENT0);
            gl::ReadPixels(
                x as GLint,
                y as GLint,
                1,
                1,
                self.color_texture
                    .as_ref()
                    .expect("Color buffer is not attached.")
                    .src_format.expect("Source format is unknown."),
                self.color_texture
                    .as_ref()
                    .expect("Color buffer is not attached.")
                    .type_.expect("Source format is unknown."),
                ptr
            );
            gl::ReadBuffer(gl::NONE);
        }
    }

    // Binder shared between instances of the FrameBuffer structure
    fn binder() -> &'static Mutex<Binder> {
        static BINDER: OnceLock<Mutex<Binder>> = OnceLock::new();
        BINDER.get_or_init(|| Mutex::new(Binder::new()))
    }
}

impl Bindable for FrameBuffer {
    fn bind(&self) -> bool {
        Self::binder().lock().unwrap().bind(self.id, |i| {
            unsafe {
                gl::BindFramebuffer(gl::FRAMEBUFFER,i);
            }
        })
    }

    fn unbind(&self) -> bool {
        Self::binder().lock().unwrap().unbind(|| {
            unsafe {
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            }
        })
    }
}

impl Drop for FrameBuffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteFramebuffers(1, &self.id);
        }
    }
}

enum Primitives {
    // TODO
}

pub trait Drawable {
    fn draw(&self);
}

#[derive(Clone)]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

pub struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Color {
    pub fn rgb(r: f32, g: f32, b: f32) -> Color {
        Color {
            r,
            g,
            b,
            a: 1.0
        }
    }
    pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color {
            r,
            g,
            b,
            a,
        }
    }
}


pub struct RenderSettings {
    // Blending
    pub blend: bool,
    pub blend_func_source_factor: GLenum,
    pub blend_func_destination_factor: GLenum,

    // Viewport
    pub viewport: Rect<GLint>,

    // Scissor
    pub scissor: Rect<GLint>,

    // Depth buffer
    pub depth_buffer: bool,

    // SRGB Framebuffer
    pub framebuffer_srgb: bool,

    // Scissor test
    pub scissor_test: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        let mut viewport: [GLint; 4] = [0, 0, 0, 0];
        unsafe {
            gl::GetIntegerv(gl::VIEWPORT, viewport.as_mut_ptr())
        }

        RenderSettings {
            blend: false,
            blend_func_source_factor: gl::SRC_ALPHA,
            blend_func_destination_factor: gl::ONE_MINUS_SRC_ALPHA,

            viewport: Rect {
                x: viewport[0],
                y: viewport[1],
                width: viewport[2],
                height: viewport[3],
            },

            scissor: Rect {
                x: viewport[0],
                y: viewport[1],
                width: viewport[2],
                height: viewport[3],
            },

            depth_buffer: false,

            framebuffer_srgb: false,

            scissor_test: false,
        }
    }
}

impl RenderSettings {
    fn set_flag(flag: bool, for_what: GLenum) {
        if flag {
            unsafe {
                gl::Enable(for_what);
            }
        } else {
            unsafe {
                gl::Disable(for_what);
            }
        }
    }

    fn set(&self) {
        unsafe {
            if self.blend {
                gl::Enable(gl::BLEND);
                gl::BlendFunc(self.blend_func_source_factor, self.blend_func_destination_factor);
            } else {
                gl::Disable(gl::BLEND);
            }

            Self::set_flag(self.depth_buffer, gl::DEPTH_TEST);
            Self::set_flag(self.framebuffer_srgb, gl::FRAMEBUFFER_SRGB);
            Self::set_flag(self.scissor_test, gl::SCISSOR_TEST);

            unsafe {
                gl::Viewport(
                    self.viewport.x,
                    self.viewport.y,
                    self.viewport.width,
                    self.viewport.height,
                );
                gl::Scissor(
                    self.scissor.x,
                    self.scissor.y,
                    self.scissor.width,
                    self.scissor.height,
                );
            }
        }
    }

    pub fn sum(&self, other: &RenderSettings) -> RenderSettings {
        RenderSettings {
            depth_buffer: self.depth_buffer || other.depth_buffer,
            framebuffer_srgb: self.framebuffer_srgb || other.framebuffer_srgb,
            viewport: self.viewport.clone(),
            blend: self.blend || other.blend,
            blend_func_source_factor: self.blend_func_source_factor,
            blend_func_destination_factor: self.blend_func_destination_factor,
            scissor_test: self.scissor_test || other.scissor_test,
            scissor: self.scissor.clone(),
        }
    }
}

pub struct RenderTarget {
    fb: FrameBuffer,
    settings: RenderSettings,
}

impl RenderTarget {
    /// Creates new on screen render target (with the handle to default framebuffer, with id 0).
    pub fn onscreen() -> RenderTarget {
        RenderTarget {
            fb: FrameBuffer::get_onscreen(),
            settings: RenderSettings::default(),
        }
    }

    /// Creates new on screen render target (with the handle to default framebuffer, with id 0) with
    /// the `settings`.
    pub fn onscreen_with_settings(settings: RenderSettings) -> RenderTarget {
        RenderTarget {
            fb: FrameBuffer::get_onscreen(),
            settings,
        }
    }

    pub fn offscreen(offscreen_fb: FrameBuffer) -> RenderTarget {
        assert!(!offscreen_fb.default, "Given framebuffer is not an offscreen framebuffer.");
        RenderTarget {
            fb: offscreen_fb,
            settings: RenderSettings::default(),
        }
    }

    pub fn offscreen_with_settings(offscreen_fb: FrameBuffer, settings: RenderSettings) -> RenderTarget {
        assert!(!offscreen_fb.default, "Given framebuffer is not an offscreen framebuffer.");
        RenderTarget {
            fb: offscreen_fb,
            settings,
        }
    }
    pub fn clear_with_color(&self, color: Color) {
        self.fb.bind();

        self.settings.set();
        unsafe {
            gl::ClearColor(color.r, color.g, color.b, color.a);
        }

        self.clear();
    }
    pub fn clear(&self) {
        self.fb.bind();

        let mut bitfield = gl::COLOR_BUFFER_BIT;
        if self.settings.depth_buffer {
            bitfield |= gl::DEPTH_BUFFER_BIT;
        }

        unsafe {

            gl::Clear(bitfield);
        }
    }

   pub fn set_blending_func(&mut self, source: GLenum, destination: GLenum) {
        self.settings.blend_func_source_factor = source;
        self.settings.blend_func_destination_factor = destination;
    }

    pub fn set_viewport(&mut self, x: GLint, y: GLint, width: GLint, height: GLint) {
        self.settings.viewport = Rect {
            x,
            y,
            width,
            height
        }

    }

    pub fn set_scissor(&mut self, x: GLint, y: GLint, width: GLint, height: GLint) {
        self.settings.scissor = Rect {
            x,
            y,
            width,
            height
        }
    }
    pub fn draw(&self, drawable_obj: &dyn Drawable, program: &ShaderProgram) {
        self.fb.bind();

        self.settings.set();
        program.bind();
        drawable_obj.draw();
    }

    pub fn draw_with_settings(&self, drawable_obj: &dyn Drawable, program: &ShaderProgram, settings: RenderSettings) {
        self.fb.bind();

        self.settings.sum(&settings).set();
        program.bind();
        drawable_obj.draw();
    }

    pub fn draw_arrays(&self, mode: GLenum, size: usize, program: &ShaderProgram) {
        self.fb.bind();

        self.settings.set();
        program.bind();
        unsafe {
            gl::DrawArrays(mode, 0, size as GLsizei);
        }
    }

    pub fn draw_arrays_with_settings(&self, mode: GLenum, size: usize, program: &ShaderProgram, settings: RenderSettings) {
        self.fb.bind();

        self.settings.sum(&settings).set();
        program.bind();
        unsafe {
            gl::DrawArrays(mode, 0, size as GLsizei);
        }
    }

    pub fn draw_elements(&self, mode: GLenum, size: usize, program: &ShaderProgram) {
        self.fb.bind();

        self.settings.set();
        program.bind();
        unsafe {
            gl::DrawElements(
                mode,
                size as GLsizei,
                gl::UNSIGNED_INT,
                core::ptr::null()
            );
        }
    }

    pub fn draw_elements_with_settings(&self, mode: GLenum, size: usize, program: &ShaderProgram, settings: RenderSettings) {
        self.fb.bind();

        self.settings.sum(&settings).set();
        program.bind();
        unsafe {
            gl::DrawElements(
                mode,
                size as GLsizei,
                gl::UNSIGNED_INT,
                core::ptr::null()
            );
        }
    }
}