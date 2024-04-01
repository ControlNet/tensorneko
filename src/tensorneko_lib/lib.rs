use pyo3::prelude::*;

pub mod evaluation;

#[pymodule]
fn tensorneko_lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // tensorneko_lib.evaluation
    let evaluation = PyModule::new_bound(m.py(), "evaluation")?;
    evaluation.add_function(wrap_pyfunction!(evaluation::ap_1d::ap_1d, &evaluation)?)?;
    evaluation.add_function(wrap_pyfunction!(evaluation::ar_1d::ar_1d, &evaluation)?)?;
    m.add_submodule(&evaluation)?;
    Ok(())
}
