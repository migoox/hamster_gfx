    use std::path::Path;
    use crate::renderer::{Shader, ShaderProgram};
    use glfw::Window;

    pub struct Painter<'a> {
        shader_program: ShaderProgram,
        glfw_window: &'a Window,
    }

    impl Painter<'_> {
        pub fn build(_glfw_window: &Window) -> Result<Painter, String> {
            let vs = Shader::compile_from_path(&Path::new("resources/egui_shaders/vertex.vert"), gl::VERTEX_SHADER)
                .expect("Painter couldn't load the vertex egui_shaders");
            let fs = Shader::compile_from_path(&Path::new("resources/egui_shader/fragment.frag"), gl::FRAGMENT_SHADER)
                .expect("Painter couldn't load the vertex egui_shaders");

            let _shader_program = ShaderProgram::link(&vs, &fs);
            todo!();
        }
    }