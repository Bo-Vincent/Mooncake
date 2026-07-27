use std::collections::{BTreeMap, BTreeSet};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::error::{ConversionError, Result};
use crate::limits::{ConversionLimits, DEFAULT_CONVERSION_LIMITS};

const CANONICAL_VIEW_KIND: &str = "canonical-affine:v1";
const MAX_DEVICE_ID: u64 = i32::MAX as u64;
const UINT64_ADDRESS_SPACE: u128 = 1_u128 << 64;

fn validation(message: impl Into<String>) -> ConversionError {
    ConversionError::validation(message)
}

fn require_nonempty(value: &str, name: &str) -> Result<()> {
    if value.is_empty() {
        return Err(validation(format!("{name} must be a non-empty string")));
    }
    Ok(())
}

fn deserialize_optional_non_null<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

fn sorted_json_value(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(sorted_json_value).collect()),
        Value::Object(values) => {
            let mut entries: Vec<_> = values.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = Map::new();
            for (name, value) in entries {
                sorted.insert(name, sorted_json_value(value));
            }
            Value::Object(sorted)
        }
        scalar => scalar,
    }
}

fn canonical_json<T: Serialize>(value: &T) -> Result<String> {
    let value = serde_json::to_value(value)?;
    Ok(serde_json::to_string(&sorted_json_value(value))?)
}

fn validate_address_range(addr: u64, storage_nbytes: u64) -> Result<()> {
    if addr == 0 {
        return Err(validation("addr must be an integer at least 1"));
    }
    if storage_nbytes == 0 {
        return Err(validation("storage_nbytes must be an integer at least 1"));
    }
    let end = u128::from(addr)
        .checked_add(u128::from(storage_nbytes))
        .ok_or_else(|| validation("addr + storage_nbytes arithmetic overflow"))?;
    if end > UINT64_ADDRESS_SPACE {
        return Err(validation(
            "addr + storage_nbytes exceeds the 64-bit address space",
        ));
    }
    Ok(())
}

fn validate_positive(values: &[u64], name: &str) -> Result<()> {
    if values.is_empty() {
        return Err(validation(format!("{name} must not be empty")));
    }
    if values.iter().any(|value| *value == 0) {
        return Err(validation(format!(
            "{name} must contain integers at least 1"
        )));
    }
    Ok(())
}

fn validate_nonempty(values: &[u64], name: &str) -> Result<()> {
    if values.is_empty() {
        return Err(validation(format!("{name} must not be empty")));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelRank {
    pub dp: u64,
    pub tp: u64,
    pub pp: u64,
    pub ep: u64,
}

impl Default for ParallelRank {
    fn default() -> Self {
        Self {
            dp: 0,
            tp: 0,
            pp: 0,
            ep: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelConfig {
    pub tp_size: u64,
    pub pp_size: u64,
    pub ep_size: u64,
    pub dp_size: u64,
}

impl ParallelConfig {
    pub fn validate(&self, limits: &ConversionLimits) -> Result<()> {
        let maximum = limits
            .max_parallel_size
            .min(DEFAULT_CONVERSION_LIMITS.max_parallel_size);
        for (name, value) in [
            ("tp_size", self.tp_size),
            ("pp_size", self.pp_size),
            ("ep_size", self.ep_size),
            ("dp_size", self.dp_size),
        ] {
            if value == 0 || u128::from(value) > maximum as u128 {
                return Err(validation(format!(
                    "{name} parallel size must be between 1 and \
                     max_parallel_size={maximum}"
                )));
            }
        }
        Ok(())
    }

    pub fn validate_rank(&self, rank: &ParallelRank) -> Result<()> {
        for (axis, value, size) in [
            ("tp", rank.tp, self.tp_size),
            ("pp", rank.pp, self.pp_size),
            ("ep", rank.ep, self.ep_size),
            ("dp", rank.dp, self.dp_size),
        ] {
            if value >= size {
                return Err(validation(format!(
                    "{axis} rank {value} is outside configured {axis}_size {size}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertSemantic {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_id: Option<u64>,
}

impl ExpertSemantic {
    pub fn validate(&self) -> Result<()> {
        require_nonempty(&self.kind, "expert kind")?;
        match self.kind.as_str() {
            "routed" if self.global_id.is_some() => Ok(()),
            "routed" => Err(validation("expert global_id must be an integer at least 0")),
            "shared" if self.global_id.is_none() => Ok(()),
            "shared" => Err(validation("shared expert must not define global_id")),
            other => Err(validation(format!("unsupported expert kind: {other}"))),
        }
    }
}

fn deserialize_required_nullable<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<u64>::deserialize(deserializer)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorSemanticWire {
    model_part: String,
    stack: String,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    layer_id: Option<u64>,
    module_path: Vec<String>,
    parameter_role: String,
    representation_id: String,
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    expert: Option<ExpertSemantic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorSemantic {
    pub model_part: String,
    pub stack: String,
    pub layer_id: Option<u64>,
    pub module_path: Vec<String>,
    pub parameter_role: String,
    pub representation_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert: Option<ExpertSemantic>,
}

impl<'de> Deserialize<'de> for TensorSemantic {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TensorSemanticWire::deserialize(deserializer)?;
        Ok(Self {
            model_part: wire.model_part,
            stack: wire.stack,
            layer_id: wire.layer_id,
            module_path: wire.module_path,
            parameter_role: wire.parameter_role,
            representation_id: wire.representation_id,
            expert: wire.expert,
        })
    }
}

impl TensorSemantic {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("model_part", self.model_part.as_str()),
            ("stack", self.stack.as_str()),
            ("parameter_role", self.parameter_role.as_str()),
            ("representation_id", self.representation_id.as_str()),
        ] {
            require_nonempty(value, name)?;
        }
        if self.module_path.is_empty() {
            return Err(validation("module_path must not be empty"));
        }
        for item in &self.module_path {
            require_nonempty(item, "module_path item")?;
        }
        if let Some(expert) = &self.expert {
            expert.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PostProcessSpec {
    pub operation: String,
    pub value: f64,
}

impl PostProcessSpec {
    fn validate(&self) -> Result<()> {
        require_nonempty(&self.operation, "postprocess operation")?;
        if self.operation != "clamp_min" {
            return Err(validation(format!(
                "unsupported postprocess operation: {}",
                self.operation
            )));
        }
        if !self.value.is_finite() {
            return Err(validation("postprocess value must be a finite number"));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeTensorManifestWire {
    tensor_id: String,
    semantic: TensorSemantic,
    global_shape: Vec<u64>,
    logical_offset: Vec<u64>,
    logical_shape: Vec<u64>,
    rank: ParallelRank,
    local_shape: Vec<u64>,
    view_kind: String,
    dtype: String,
    itemsize: u64,
    strides_bytes: Vec<u64>,
    dev: u64,
    addr: u64,
    storage_nbytes: u64,
    #[serde(default)]
    fragment_id: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    endpoint: Option<String>,
    #[serde(default)]
    worker_id: Option<String>,
    #[serde(default)]
    lease_generation: Option<u64>,
    #[serde(default)]
    postprocess: Vec<PostProcessSpec>,
    #[serde(
        default,
        rename = "semantic_fingerprint",
        deserialize_with = "deserialize_optional_non_null"
    )]
    legacy_semantic_fingerprint: Option<String>,
    #[serde(
        default,
        rename = "physical_layout_id",
        deserialize_with = "deserialize_optional_non_null"
    )]
    legacy_physical_layout_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RuntimeTensorManifest {
    pub tensor_id: String,
    pub semantic: TensorSemantic,
    pub global_shape: Vec<u64>,
    pub logical_offset: Vec<u64>,
    pub logical_shape: Vec<u64>,
    pub rank: ParallelRank,
    pub local_shape: Vec<u64>,
    pub view_kind: String,
    pub dtype: String,
    pub itemsize: u64,
    pub strides_bytes: Vec<u64>,
    pub dev: u64,
    pub addr: u64,
    pub storage_nbytes: u64,
    #[serde(skip_serializing)]
    pub fragment_id: Option<String>,
    #[serde(skip_serializing)]
    pub worker_id: Option<String>,
    #[serde(skip_serializing)]
    pub lease_generation: Option<u64>,
    #[serde(skip_serializing)]
    pub postprocess: Vec<PostProcessSpec>,
    #[serde(skip_serializing)]
    pub legacy_semantic_fingerprint: Option<String>,
    #[serde(skip_serializing)]
    pub legacy_physical_layout_id: Option<String>,
}

impl<'de> Deserialize<'de> for RuntimeTensorManifest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RuntimeTensorManifestWire::deserialize(deserializer)?;
        if let Some(endpoint) = &wire.endpoint {
            let expected = format!("dev:{}", wire.dev);
            if endpoint != &expected {
                return Err(D::Error::custom(
                    "legacy endpoint does not match global dev",
                ));
            }
        }
        Ok(Self {
            tensor_id: wire.tensor_id,
            semantic: wire.semantic,
            global_shape: wire.global_shape,
            logical_offset: wire.logical_offset,
            logical_shape: wire.logical_shape,
            rank: wire.rank,
            local_shape: wire.local_shape,
            view_kind: wire.view_kind,
            dtype: wire.dtype,
            itemsize: wire.itemsize,
            strides_bytes: wire.strides_bytes,
            dev: wire.dev,
            addr: wire.addr,
            storage_nbytes: wire.storage_nbytes,
            fragment_id: wire.fragment_id,
            worker_id: wire.worker_id,
            lease_generation: wire.lease_generation,
            postprocess: wire.postprocess,
            legacy_semantic_fingerprint: wire.legacy_semantic_fingerprint,
            legacy_physical_layout_id: wire.legacy_physical_layout_id,
        })
    }
}

impl RuntimeTensorManifest {
    pub fn validate(&self, limits: &ConversionLimits) -> Result<()> {
        require_nonempty(&self.tensor_id, "tensor_id")?;
        self.semantic.validate()?;
        if self.view_kind != CANONICAL_VIEW_KIND {
            return Err(validation(format!(
                "view_kind must be {CANONICAL_VIEW_KIND}; got {:?}",
                self.view_kind
            )));
        }
        require_nonempty(&self.dtype, "dtype")?;
        if self.itemsize == 0 {
            return Err(validation("itemsize must be an integer at least 1"));
        }
        if self.dev > MAX_DEVICE_ID {
            return Err(validation(format!(
                "dev exceeds the supported maximum {MAX_DEVICE_ID}"
            )));
        }
        validate_address_range(self.addr, self.storage_nbytes)?;

        validate_positive(&self.global_shape, "global_shape")?;
        validate_nonempty(&self.logical_offset, "logical_offset")?;
        validate_positive(&self.logical_shape, "logical_shape")?;
        validate_positive(&self.local_shape, "local_shape")?;
        validate_positive(&self.strides_bytes, "strides_bytes")?;

        let ndim = self.global_shape.len();
        if ndim > limits.max_tensor_ndim {
            return Err(validation(format!(
                "tensor rank exceeds max_tensor_ndim={}",
                limits.max_tensor_ndim
            )));
        }
        if self.logical_offset.len() != ndim
            || self.logical_shape.len() != ndim
            || self.local_shape.len() != ndim
            || self.strides_bytes.len() != ndim
        {
            return Err(validation(
                "tensor logical and physical geometry rank mismatch",
            ));
        }

        for ((begin, extent), total) in self
            .logical_offset
            .iter()
            .zip(&self.logical_shape)
            .zip(&self.global_shape)
        {
            let end = begin.checked_add(*extent).ok_or_else(|| {
                validation(format!(
                    "logical box arithmetic overflow: {}",
                    self.tensor_id
                ))
            })?;
            if end > *total {
                return Err(validation(format!(
                    "logical box exceeds global shape: {}",
                    self.tensor_id
                )));
            }
        }
        if self
            .logical_shape
            .iter()
            .zip(&self.local_shape)
            .any(|(valid, physical)| valid > physical)
        {
            return Err(validation("logical_shape must fit within local_shape"));
        }
        if self.strides_bytes.last() != Some(&self.itemsize) {
            return Err(validation("last local dimension must be contiguous"));
        }
        if self
            .strides_bytes
            .iter()
            .any(|stride| stride % self.itemsize != 0)
        {
            return Err(validation("strides_bytes must be multiples of itemsize"));
        }

        let mut storage_span = u128::from(self.itemsize);
        for (extent, stride) in self.local_shape.iter().zip(&self.strides_bytes) {
            let contribution = u128::from(*extent - 1)
                .checked_mul(u128::from(*stride))
                .ok_or_else(|| validation("local storage span arithmetic overflow"))?;
            storage_span = storage_span
                .checked_add(contribution)
                .ok_or_else(|| validation("local storage span arithmetic overflow"))?;
        }
        if storage_span > u128::from(self.storage_nbytes) {
            return Err(validation(format!(
                "local storage span {storage_span} exceeds storage_nbytes {}: {}",
                self.storage_nbytes, self.tensor_id
            )));
        }

        if let Some(fragment_id) = &self.fragment_id {
            require_nonempty(fragment_id, "fragment_id")?;
        }
        if let Some(worker_id) = &self.worker_id {
            require_nonempty(worker_id, "worker_id")?;
        }
        if let Some(semantic_fingerprint) = &self.legacy_semantic_fingerprint {
            require_nonempty(semantic_fingerprint, "semantic_fingerprint")?;
            if semantic_fingerprint != &self.semantic.representation_id {
                return Err(validation(
                    "legacy semantic_fingerprint does not match representation_id",
                ));
            }
        }
        if let Some(physical_layout_id) = &self.legacy_physical_layout_id {
            require_nonempty(physical_layout_id, "physical_layout_id")?;
            if physical_layout_id != &self.view_kind {
                return Err(validation(
                    "legacy physical_layout_id does not match canonical view_kind",
                ));
            }
        }
        for operation in &self.postprocess {
            operation.validate()?;
        }
        if !self.postprocess.is_empty() {
            return Err(validation(
                "postprocess is unsupported by the copy-only sCR interface",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelManifest {
    pub model_id: String,
    pub parallel: ParallelConfig,
    pub tensors: Vec<RuntimeTensorManifest>,
    #[serde(default, skip_serializing)]
    pub revision: Option<String>,
    #[serde(default, skip_serializing)]
    pub instance_id: Option<String>,
    #[serde(default, skip_serializing)]
    pub generation: Option<u64>,
}

impl RuntimeModelManifest {
    pub fn validate(&self, limits: &ConversionLimits) -> Result<()> {
        require_nonempty(&self.model_id, "model_id")?;
        self.parallel.validate(limits)?;
        if self.tensors.is_empty() {
            return Err(validation("runtime manifest tensors must not be empty"));
        }
        if self.tensors.len() > limits.max_tensor_records {
            return Err(validation(format!(
                "runtime manifest tensor record count exceeds \
                 max_tensor_records={}",
                limits.max_tensor_records
            )));
        }
        for tensor in &self.tensors {
            tensor.validate(limits)?;
            self.parallel.validate_rank(&tensor.rank)?;
        }
        if let Some(revision) = &self.revision {
            require_nonempty(revision, "revision")?;
        }
        if let Some(instance_id) = &self.instance_id {
            require_nonempty(instance_id, "instance_id")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConversionRequest {
    pub format_version: u64,
    pub plan_id: String,
    pub source_manifest: RuntimeModelManifest,
    pub target_manifest: RuntimeModelManifest,
}

impl ConversionRequest {
    pub fn from_json(value: &str) -> Result<Self> {
        Self::from_json_with_limits(value, &DEFAULT_CONVERSION_LIMITS)
    }

    pub fn from_json_with_limits(value: &str, limits: &ConversionLimits) -> Result<Self> {
        limits.validate()?;
        let encoded_size = value.len();
        if encoded_size > limits.max_json_bytes {
            return Err(validation(format!(
                "conversion request JSON exceeds max_json_bytes={}",
                limits.max_json_bytes
            )));
        }
        let request: Self = serde_json::from_str(value)?;
        request.validate_with_limits(limits)?;
        Ok(request)
    }

    pub fn to_json(&self) -> Result<String> {
        canonical_json(self)
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_with_limits(&DEFAULT_CONVERSION_LIMITS)
    }

    pub fn validate_with_limits(&self, limits: &ConversionLimits) -> Result<()> {
        limits.validate()?;
        if self.format_version != 1 {
            return Err(validation(format!(
                "unsupported format_version: {}; expected 1",
                self.format_version
            )));
        }
        require_nonempty(&self.plan_id, "plan_id")?;
        if !self
            .plan_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(validation("plan_id contains unsupported key characters"));
        }

        self.source_manifest.validate(limits)?;
        self.target_manifest.validate(limits)?;
        let total_tensor_records = self
            .source_manifest
            .tensors
            .len()
            .checked_add(self.target_manifest.tensors.len())
            .ok_or_else(|| validation("conversion request tensor record count overflow"))?;
        if total_tensor_records > limits.max_tensor_records {
            return Err(validation(format!(
                "conversion request tensor record count exceeds \
                 max_tensor_records={}",
                limits.max_tensor_records
            )));
        }
        if self.source_manifest.model_id != self.target_manifest.model_id {
            return Err(validation(
                "model identity mismatch between source and target",
            ));
        }

        let mut target_dps: BTreeMap<&str, BTreeSet<u64>> = BTreeMap::new();
        for tensor in &self.target_manifest.tensors {
            target_dps
                .entry(tensor.tensor_id.as_str())
                .or_default()
                .insert(tensor.rank.dp);
        }
        for (tensor_id, actual_dps) in target_dps {
            let missing: Vec<_> = (0..self.target_manifest.parallel.dp_size)
                .filter(|dp| !actual_dps.contains(dp))
                .collect();
            if !missing.is_empty() {
                return Err(validation(format!(
                    "missing target DP replicas for tensor {tensor_id}: {missing:?}"
                )));
            }
        }
        Ok(())
    }
}
