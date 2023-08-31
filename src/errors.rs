use gl::types::GLenum;

pub fn check_opengl_errors() {
    let mut err_codes: Vec<i32> = Vec::new();
    loop {
        unsafe {
            let err: GLenum = gl::GetError();

            if err == gl::NO_ERROR {
                if err_codes.len() > 0 {
                    let s: String = err_codes
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");

                    panic!("OpenGL error codes: {}", s);
                }
                return;
            }
            err_codes.push(err as i32);
        }
    }
}