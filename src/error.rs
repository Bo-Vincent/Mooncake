use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConversionError {
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Validation(String),
}

impl ConversionError {
    pub(crate) fn validation(message: impl Into<String>) -> Self {
        Self::Validation(message.into())
    }
}

pub type Result<T> = std::result::Result<T, ConversionError>;
