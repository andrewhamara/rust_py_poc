use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyBytes;
use std::fs;

fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        // modify sys path
        let sys = py.import("sys")?;
        let sys_path: &PyList = sys.getattr("path")?.downcast()?;

        // change this to your conda env site packages
        sys_path.insert(0, "/Users/andrewhamara/miniconda3/envs/rvt/lib/python3.9/site-packages")?;
        sys_path.insert(1, "./eTraM/rvt_eTram")?;  // Add your inference.py directory

        for path in sys_path.iter() {
            println!("sys.path: {}", path);
        }

        // read inference code
        let inference_code = fs::read_to_string("./eTraM/rvt_eTram/inference.py")
            .expect("Failed to read inference.py");

        let inference_module = PyModule::from_code(py, &inference_code, "inference.py", "inference")?;

        // dummy bytes
        let dummy_event_bytes: Vec<u8> = vec![0; 2000];
        let py_byte_buffer = PyBytes::new(py, &dummy_event_bytes);

        // forward pass
        inference_module.getattr("main")?.call1((py_byte_buffer,))?;

        Ok(())
    })
}