use pyo3::prelude::*;

pub mod loc_1d;


#[pymodule]
pub fn evaluation(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(loc_1d::ap_1d, m)?)?;
    m.add_function(wrap_pyfunction!(loc_1d::ar_1d, m)?)?;
    m.add_function(wrap_pyfunction!(loc_1d::ap_ar_1d, m)?)?;
    Ok(())
}