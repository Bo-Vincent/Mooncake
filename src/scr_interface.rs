use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::error::{ConversionError, Result};
use crate::limits::{ConversionLimits, DEFAULT_CONVERSION_LIMITS};

const FORMAT_VERSION: u32 = 1;
const MAX_DEVICE_ID: u64 = i32::MAX as u64;
const UINT64_ADDRESS_LIMIT: u128 = 1_u128 << 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceMemoryLocation {
    pub dev: u64,
    pub addr: u64,
}

impl DeviceMemoryLocation {
    pub fn new(dev: u64, addr: u64) -> Result<Self> {
        let location = Self { dev, addr };
        location.validate()?;
        Ok(location)
    }

    fn validate(&self) -> Result<()> {
        if self.dev > MAX_DEVICE_ID {
            return Err(ConversionError::validation(format!(
                "dev must be at most {MAX_DEVICE_ID}"
            )));
        }
        if self.addr == 0 {
            return Err(ConversionError::validation(
                "addr must be a positive integer",
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDeviceMemoryLocation {
    dev: u64,
    addr: u64,
}

impl<'de> Deserialize<'de> for DeviceMemoryLocation {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawDeviceMemoryLocation::deserialize(deserializer)?;
        DeviceMemoryLocation::new(raw.dev, raw.addr).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScrTransferTask {
    pub key: String,
    pub tensor_id: String,
    pub source: DeviceMemoryLocation,
    pub target: DeviceMemoryLocation,
    pub nbytes: u64,
}

impl ScrTransferTask {
    pub fn new(
        key: impl Into<String>,
        tensor_id: impl Into<String>,
        source: DeviceMemoryLocation,
        target: DeviceMemoryLocation,
        nbytes: u64,
    ) -> Result<Self> {
        let task = Self {
            key: key.into(),
            tensor_id: tensor_id.into(),
            source,
            target,
            nbytes,
        };
        task.validate()?;
        Ok(task)
    }

    fn validate(&self) -> Result<()> {
        if self.key.is_empty() {
            return Err(ConversionError::validation(
                "key must be a non-empty string",
            ));
        }
        if self.tensor_id.is_empty() {
            return Err(ConversionError::validation(
                "tensor_id must be a non-empty string",
            ));
        }
        self.source.validate()?;
        self.target.validate()?;
        if self.nbytes == 0 {
            return Err(ConversionError::validation(
                "nbytes must be a positive integer",
            ));
        }

        let source_end = u128::from(self.source.addr) + u128::from(self.nbytes);
        let target_end = u128::from(self.target.addr) + u128::from(self.nbytes);
        if source_end > UINT64_ADDRESS_LIMIT || target_end > UINT64_ADDRESS_LIMIT {
            return Err(ConversionError::validation(
                "transfer range exceeds the 64-bit address space",
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawScrTransferTask {
    key: String,
    tensor_id: String,
    source: DeviceMemoryLocation,
    target: DeviceMemoryLocation,
    nbytes: u64,
}

impl<'de> Deserialize<'de> for ScrTransferTask {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawScrTransferTask::deserialize(deserializer)?;
        ScrTransferTask::new(raw.key, raw.tensor_id, raw.source, raw.target, raw.nbytes)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScrTransferPlan {
    pub format_version: u32,
    pub plan_id: String,
    pub transfers: Vec<ScrTransferTask>,
}

impl ScrTransferPlan {
    pub fn new(
        plan_id: impl Into<String>,
        transfers: Vec<ScrTransferTask>,
        limits: &ConversionLimits,
    ) -> Result<Self> {
        let plan = Self {
            format_version: FORMAT_VERSION,
            plan_id: plan_id.into(),
            transfers,
        };
        plan.validate_with_limits(limits)?;
        Ok(plan)
    }

    pub fn from_json(value: impl AsRef<[u8]>) -> Result<Self> {
        Self::from_json_with_limits(value, &DEFAULT_CONVERSION_LIMITS)
    }

    pub fn from_json_with_limits(
        value: impl AsRef<[u8]>,
        limits: &ConversionLimits,
    ) -> Result<Self> {
        limits.validate()?;
        let bytes = value.as_ref();
        if bytes.len() > limits.max_json_bytes {
            return Err(ConversionError::validation(format!(
                "sCR transfer plan exceeds max_json_bytes={}",
                limits.max_json_bytes
            )));
        }

        let plan: Self = serde_json::from_slice(bytes)?;
        plan.validate_with_limits(limits)?;
        Ok(plan)
    }

    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        let value = serde_json::to_value(self)?;
        Ok(serde_json::to_string(&canonicalize_json(value))?)
    }

    pub fn validate_with_limits(&self, limits: &ConversionLimits) -> Result<()> {
        limits.validate()?;
        self.validate()?;
        if self.transfers.len() > limits.max_transfer_tasks {
            return Err(ConversionError::validation(format!(
                "transfer task count exceeds max_transfer_tasks={}",
                limits.max_transfer_tasks
            )));
        }
        for transfer in &self.transfers {
            if transfer.key.len() > limits.max_key_bytes {
                return Err(ConversionError::validation(format!(
                    "transfer key exceeds max_key_bytes={}",
                    limits.max_key_bytes
                )));
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.format_version != FORMAT_VERSION {
            return Err(ConversionError::validation(format!(
                "unsupported format_version: {}; expected {FORMAT_VERSION}",
                self.format_version
            )));
        }
        if !is_key_component(&self.plan_id) {
            return Err(ConversionError::validation(
                "plan_id contains unsupported key characters",
            ));
        }

        let mut keys = BTreeSet::new();
        for transfer in &self.transfers {
            transfer.validate()?;
            if !keys.insert(transfer.key.as_str()) {
                return Err(ConversionError::validation(
                    "sCR transfer plan contains duplicate keys",
                ));
            }
            if !key_is_bound_to_plan(&transfer.key, &self.plan_id) {
                return Err(ConversionError::validation(format!(
                    "transfer key is not bound to plan_id {}: {}",
                    self.plan_id, transfer.key
                )));
            }
        }

        let targets_by_dev = self.validated_targets_by_dev()?;
        self.validate_source_target_hazards(&targets_by_dev)
    }

    fn validated_targets_by_dev(&self) -> Result<BTreeMap<u64, Vec<Interval>>> {
        let mut targets_by_dev: BTreeMap<u64, Vec<Interval>> = BTreeMap::new();
        for (index, transfer) in self.transfers.iter().enumerate() {
            targets_by_dev
                .entry(transfer.target.dev)
                .or_default()
                .push(Interval {
                    begin: u128::from(transfer.target.addr),
                    end: u128::from(transfer.target.addr) + u128::from(transfer.nbytes),
                    task_index: index,
                });
        }

        for intervals in targets_by_dev.values_mut() {
            intervals.sort_unstable_by_key(|item| (item.begin, item.end, item.task_index));
            for pair in intervals.windows(2) {
                if pair[1].begin < pair[0].end {
                    return Err(ConversionError::validation(
                        "sCR transfer plan has overlapping target writes",
                    ));
                }
            }
        }
        Ok(targets_by_dev)
    }

    fn validate_source_target_hazards(
        &self,
        targets_by_dev: &BTreeMap<u64, Vec<Interval>>,
    ) -> Result<()> {
        for (source_index, transfer) in self.transfers.iter().enumerate() {
            let Some(targets) = targets_by_dev.get(&transfer.source.dev) else {
                continue;
            };
            let source_begin = u128::from(transfer.source.addr);
            let source_end = source_begin + u128::from(transfer.nbytes);
            let position = targets.partition_point(|target| target.begin <= source_begin);

            for candidate_index in [position.checked_sub(1), Some(position)]
                .into_iter()
                .flatten()
            {
                let Some(target) = targets.get(candidate_index) else {
                    continue;
                };
                if source_begin >= target.end || target.begin >= source_end {
                    continue;
                }
                let same_noop = source_index == target.task_index
                    && source_begin == target.begin
                    && source_end == target.end;
                if !same_noop {
                    return Err(ConversionError::validation(
                        "sCR transfer plan has a source-target physical hazard",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawScrTransferPlan {
    format_version: u32,
    plan_id: String,
    transfers: Vec<ScrTransferTask>,
}

impl<'de> Deserialize<'de> for ScrTransferPlan {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawScrTransferPlan::deserialize(deserializer)?;
        let plan = Self {
            format_version: raw.format_version,
            plan_id: raw.plan_id,
            transfers: raw.transfers,
        };
        plan.validate().map_err(serde::de::Error::custom)?;
        Ok(plan)
    }
}

#[derive(Debug, Clone, Copy)]
struct Interval {
    begin: u128,
    end: u128,
    task_index: usize,
}

fn is_key_component(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn is_decimal(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

fn valid_rank_component(value: &str, prefix: &str) -> bool {
    let Some(value) = value.strip_prefix(prefix) else {
        return false;
    };
    let Some((pp, remainder)) = value.split_once('t') else {
        return false;
    };
    let Some((tp, ep)) = remainder.split_once('e') else {
        return false;
    };
    is_decimal(pp) && is_decimal(tp) && is_decimal(ep)
}

fn key_is_bound_to_plan(key: &str, plan_id: &str) -> bool {
    let prefix = format!("wt/tx-{plan_id}-");
    let Some(key) = key.strip_prefix(&prefix) else {
        return false;
    };
    let mut path = key.split('/');
    let (Some(operation), Some(source), Some(target), None) =
        (path.next(), path.next(), path.next(), path.next())
    else {
        return false;
    };

    let mut operation_parts = operation.rsplitn(3, '-');
    let (Some(chunk), Some(segment), Some(operation_id)) = (
        operation_parts.next(),
        operation_parts.next(),
        operation_parts.next(),
    ) else {
        return false;
    };

    is_key_component(operation_id)
        && is_decimal(segment)
        && is_decimal(chunk)
        && valid_rank_component(source, "S-p")
        && valid_rank_component(target, "T-p")
}

fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.into_iter().map(canonicalize_json).collect()),
        Value::Object(items) => {
            let sorted: BTreeMap<_, _> = items
                .into_iter()
                .map(|(key, value)| (key, canonicalize_json(value)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        scalar => scalar,
    }
}
