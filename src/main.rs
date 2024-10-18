use pyo3::prelude::*;
use pyo3::types::{PyList,PyBytes};
use std::fs;

fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    // read the inference.py file
    let inference_code = fs::read_to_string("./eTraM/rvt_eTram/inference.py")
        .expect("Failed to read inference.py");

    Python::with_gil(|py| -> PyResult<()> {

        let sys = py.import("sys")?;
        let sys_path: &PyList = sys.getattr("path")?.downcast()?;

        // add the path to your Conda environment and the directory where torch is installed
        sys_path.insert(0, "/Users/andrewhamara/miniconda3/envs/rvt/lib/python3.9/site-packages")?;
        sys_path.insert(1, "./eTraM/rvt_eTram")?;  // Add your inference.py directory

        // load inference module
        let inference_module = PyModule::from_code(py, &inference_code, "inference.py", "inference")?;

        // dummy bytes
        let dummy_event_bytes: Vec<u8> = vec![0; 2000];
        let py_event_bytes = PyBytes::new(py, &dummy_event_bytes);

        // forward pass with no LSTM states
        let results: (Vec<usize>, &PyBytes) = inference_module.getattr("main")?
            .call1((py_event_bytes, None::<Option<PyObject>>))?
            .extract()?;

        let (predictions, hidden_states): (Vec<usize>, &PyBytes) = results;

        // print predictions
        println!("Predictions from Python: {:?}", predictions);

        // LSTM states
        let serialized_hidden_states = hidden_states.to_object(py);

        // call again using hidden states from first pass
        let results_next: (Vec<usize>, &PyBytes) = inference_module.getattr("main")?
            .call1((py_event_bytes, serialized_hidden_states))?
            .extract()?;

        let (next_predictions, _next_hidden_states) = results_next;

        println!("Next Predictions: {:?}", next_predictions);

        Ok(())
    })
}