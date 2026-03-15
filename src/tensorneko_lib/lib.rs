use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

pub mod evaluation;

#[pymodule]
fn tensorneko_lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // tensorneko_lib.evaluation
    m.add_wrapped(wrap_pymodule!(evaluation::evaluation))?;
    let sys = PyModule::import(_py, "sys")?;
    let sys_module: Bound<'_, PyDict> = sys.getattr("modules")?.cast_into_exact()?;
    sys_module.set_item("tensorneko_lib.evaluation", m.getattr("evaluation")?)?;
    Ok(())
}
