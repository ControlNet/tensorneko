use pyo3::prelude::*;

pub mod ap_1d;
pub mod ar_1d;


#[pymodule]
pub fn evaluation(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ar_1d::ar_1d, m)?)?;
    m.add_function(wrap_pyfunction!(ap_1d::ap_1d, m)?)?;
    Ok(())
}