use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("Model has already been trained")]
    ModelAlreadyTrained,

    #[error("Model has not been trained")]
    ModelNotTrained,
}
