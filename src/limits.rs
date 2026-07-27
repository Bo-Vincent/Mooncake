use crate::error::{ConversionError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConversionLimits {
    pub max_json_bytes: usize,
    pub max_tensor_records: usize,
    pub max_tensor_ndim: usize,
    pub max_parallel_size: usize,
    pub max_partition_cells: usize,
    pub max_candidate_assignments: usize,
    pub max_copy_operations: usize,
    pub max_transfer_tasks: usize,
    pub max_key_bytes: usize,
    pub max_staging_chunk_bytes: usize,
}

pub const DEFAULT_CONVERSION_LIMITS: ConversionLimits = ConversionLimits {
    max_json_bytes: 64 * 1024 * 1024,
    max_tensor_records: 1_000_000,
    max_tensor_ndim: 8,
    max_parallel_size: 65_536,
    max_partition_cells: 250_000,
    max_candidate_assignments: 10_000_000,
    max_copy_operations: 500_000,
    max_transfer_tasks: 1_000_000,
    max_key_bytes: 512,
    max_staging_chunk_bytes: 64 * 1024 * 1024,
};

impl Default for ConversionLimits {
    fn default() -> Self {
        DEFAULT_CONVERSION_LIMITS
    }
}

impl ConversionLimits {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("max_json_bytes", self.max_json_bytes),
            ("max_tensor_records", self.max_tensor_records),
            ("max_tensor_ndim", self.max_tensor_ndim),
            ("max_parallel_size", self.max_parallel_size),
            ("max_partition_cells", self.max_partition_cells),
            ("max_candidate_assignments", self.max_candidate_assignments),
            ("max_copy_operations", self.max_copy_operations),
            ("max_transfer_tasks", self.max_transfer_tasks),
            ("max_key_bytes", self.max_key_bytes),
            ("max_staging_chunk_bytes", self.max_staging_chunk_bytes),
        ] {
            if value == 0 {
                return Err(ConversionError::validation(format!(
                    "{name} must be a positive integer"
                )));
            }
        }
        Ok(())
    }
}
