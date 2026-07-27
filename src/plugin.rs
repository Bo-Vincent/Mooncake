use crate::error::{ConversionError, Result};
use crate::limits::{ConversionLimits, DEFAULT_CONVERSION_LIMITS};
use crate::manifest::{ConversionRequest, RuntimeModelManifest};
use crate::planner::plan_scr_transfer;
use crate::scr_interface::ScrTransferPlan;

#[derive(Debug, Clone)]
pub struct ManifestWeightConversionPlugin {
    limits: ConversionLimits,
}

impl Default for ManifestWeightConversionPlugin {
    fn default() -> Self {
        Self {
            limits: DEFAULT_CONVERSION_LIMITS,
        }
    }
}

impl ManifestWeightConversionPlugin {
    pub fn new(limits: ConversionLimits) -> Result<Self> {
        limits.validate()?;
        Ok(Self { limits })
    }

    pub fn limits(&self) -> &ConversionLimits {
        &self.limits
    }

    pub fn plan_scr(
        &self,
        request: &ConversionRequest,
        max_chunk_bytes: Option<usize>,
    ) -> Result<ScrTransferPlan> {
        plan_scr_transfer(request, max_chunk_bytes, &self.limits)
    }

    pub fn plan_scr_json(&self, request: &str, max_chunk_bytes: Option<usize>) -> Result<String> {
        let request = ConversionRequest::from_json_with_limits(request, &self.limits)?;
        self.plan_scr(&request, max_chunk_bytes)?.to_json()
    }

    pub fn validate_scr_transfer_plan(
        &self,
        request: &ConversionRequest,
        plan: &ScrTransferPlan,
        max_chunk_bytes: Option<usize>,
    ) -> Result<()> {
        let expected = self.plan_scr(request, max_chunk_bytes)?;
        if plan != &expected {
            return Err(ConversionError::validation(
                "sCR transfer plan does not match conversion request",
            ));
        }
        Ok(())
    }
}

pub fn create_plan(
    plan_id: impl Into<String>,
    source_manifest: RuntimeModelManifest,
    target_manifest: RuntimeModelManifest,
    max_chunk_bytes: Option<usize>,
) -> Result<ScrTransferPlan> {
    let request = ConversionRequest {
        format_version: 1,
        plan_id: plan_id.into(),
        source_manifest,
        target_manifest,
    };
    ManifestWeightConversionPlugin::default().plan_scr(&request, max_chunk_bytes)
}

pub fn create_plan_with_limits(
    plan_id: impl Into<String>,
    source_manifest: RuntimeModelManifest,
    target_manifest: RuntimeModelManifest,
    max_chunk_bytes: Option<usize>,
    limits: ConversionLimits,
) -> Result<ScrTransferPlan> {
    let request = ConversionRequest {
        format_version: 1,
        plan_id: plan_id.into(),
        source_manifest,
        target_manifest,
    };
    ManifestWeightConversionPlugin::new(limits)?.plan_scr(&request, max_chunk_bytes)
}
