use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::error::{ConversionError, Result};
use crate::limits::ConversionLimits;
use crate::manifest::{ConversionRequest, ParallelRank, RuntimeTensorManifest};
use crate::scr_interface::{DeviceMemoryLocation, ScrTransferPlan, ScrTransferTask};

fn validation(message: impl Into<String>) -> ConversionError {
    ConversionError::validation(message)
}

#[derive(Debug)]
struct PlanningBudget<'a> {
    limits: &'a ConversionLimits,
    partition_cells: usize,
    candidate_assignments: usize,
}

impl<'a> PlanningBudget<'a> {
    fn new(limits: &'a ConversionLimits) -> Self {
        Self {
            limits,
            partition_cells: 0,
            candidate_assignments: 0,
        }
    }

    fn consume_partition_cells(&mut self, count: usize) -> Result<()> {
        self.partition_cells = self
            .partition_cells
            .checked_add(count)
            .ok_or_else(|| validation("partition cell count overflow"))?;
        if self.partition_cells > self.limits.max_partition_cells {
            return Err(validation(format!(
                "partition cell count exceeds request-wide max_partition_cells={}",
                self.limits.max_partition_cells
            )));
        }
        Ok(())
    }

    fn consume_candidate_assignments(&mut self, count: usize) -> Result<()> {
        self.candidate_assignments = self
            .candidate_assignments
            .checked_add(count)
            .ok_or_else(|| validation("candidate assignment count overflow"))?;
        if self.candidate_assignments > self.limits.max_candidate_assignments {
            return Err(validation(format!(
                "candidate assignment count exceeds request-wide \
                 max_candidate_assignments={}",
                self.limits.max_candidate_assignments
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Fragment<'a> {
    id: String,
    tensor: &'a RuntimeTensorManifest,
}

fn rank_cmp(left: &ParallelRank, right: &ParallelRank) -> Ordering {
    (left.dp, left.pp, left.ep, left.tp).cmp(&(right.dp, right.pp, right.ep, right.tp))
}

fn runtime_fragment_cmp(left: &&RuntimeTensorManifest, right: &&RuntimeTensorManifest) -> Ordering {
    rank_cmp(&left.rank, &right.rank)
        .then_with(|| left.logical_offset.cmp(&right.logical_offset))
        .then_with(|| left.logical_shape.cmp(&right.logical_shape))
        .then_with(|| left.dev.cmp(&right.dev))
        .then_with(|| left.addr.cmp(&right.addr))
        .then_with(|| left.local_shape.cmp(&right.local_shape))
        .then_with(|| left.strides_bytes.cmp(&right.strides_bytes))
        .then_with(|| left.storage_nbytes.cmp(&right.storage_nbytes))
}

fn source_cmp(left: &Fragment<'_>, right: &Fragment<'_>) -> Ordering {
    rank_cmp(&left.tensor.rank, &right.tensor.rank)
        .then_with(|| format!("dev:{}", left.tensor.dev).cmp(&format!("dev:{}", right.tensor.dev)))
        .then_with(|| left.tensor.logical_offset.cmp(&right.tensor.logical_offset))
        .then_with(|| left.tensor.logical_shape.cmp(&right.tensor.logical_shape))
        .then_with(|| left.tensor.addr.cmp(&right.tensor.addr))
        .then_with(|| left.tensor.strides_bytes.cmp(&right.tensor.strides_bytes))
        .then_with(|| left.tensor.storage_nbytes.cmp(&right.tensor.storage_nbytes))
        .then_with(|| left.id.cmp(&right.id))
}

fn group_fragments<'a>(
    side: &str,
    tensors: &'a [RuntimeTensorManifest],
) -> Result<BTreeMap<String, Vec<Fragment<'a>>>> {
    let mut grouped: BTreeMap<String, Vec<&RuntimeTensorManifest>> = BTreeMap::new();
    for tensor in tensors {
        grouped
            .entry(tensor.tensor_id.clone())
            .or_default()
            .push(tensor);
    }

    let mut result = BTreeMap::new();
    for (tensor_id, mut group) in grouped {
        group.sort_by(runtime_fragment_cmp);
        let first = group
            .first()
            .copied()
            .ok_or_else(|| validation("tensor group must not be empty"))?;
        for item in group.iter().skip(1) {
            if item.semantic != first.semantic {
                return Err(validation(format!(
                    "semantic mismatch within tensor: {tensor_id}"
                )));
            }
            if item.global_shape != first.global_shape {
                return Err(validation(format!(
                    "global shape mismatch within tensor: {tensor_id}"
                )));
            }
            if item.dtype != first.dtype || item.itemsize != first.itemsize {
                return Err(validation(format!(
                    "dtype or itemsize mismatch within tensor: {tensor_id}"
                )));
            }
        }
        let fragments = group
            .into_iter()
            .enumerate()
            .map(|(index, tensor)| Fragment {
                id: format!("{side}-{tensor_id}-{index:08}"),
                tensor,
            })
            .collect();
        result.insert(tensor_id, fragments);
    }
    Ok(result)
}

fn intersection(
    left_offset: &[u64],
    left_shape: &[u64],
    right_offset: &[u64],
    right_shape: &[u64],
) -> Result<Option<(Vec<u64>, Vec<u64>)>> {
    let mut offset = Vec::with_capacity(left_shape.len());
    let mut shape = Vec::with_capacity(left_shape.len());
    for (((left_begin, left_extent), right_begin), right_extent) in left_offset
        .iter()
        .zip(left_shape)
        .zip(right_offset)
        .zip(right_shape)
    {
        let begin = (*left_begin).max(*right_begin);
        let left_end = left_begin
            .checked_add(*left_extent)
            .ok_or_else(|| validation("logical box arithmetic overflow"))?;
        let right_end = right_begin
            .checked_add(*right_extent)
            .ok_or_else(|| validation("logical box arithmetic overflow"))?;
        let end = left_end.min(right_end);
        if end <= begin {
            return Ok(None);
        }
        offset.push(begin);
        shape.push(end - begin);
    }
    Ok(Some((offset, shape)))
}

fn visit_coordinates<F>(ranges: &[std::ops::Range<usize>], mut visitor: F) -> Result<()>
where
    F: FnMut(&[usize]) -> Result<()>,
{
    fn recurse<F>(
        ranges: &[std::ops::Range<usize>],
        dim: usize,
        coordinate: &mut Vec<usize>,
        visitor: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&[usize]) -> Result<()>,
    {
        if dim == ranges.len() {
            return visitor(coordinate);
        }
        for value in ranges[dim].clone() {
            coordinate.push(value);
            recurse(ranges, dim + 1, coordinate, visitor)?;
            coordinate.pop();
        }
        Ok(())
    }

    recurse(
        ranges,
        0,
        &mut Vec::with_capacity(ranges.len()),
        &mut visitor,
    )
}

#[derive(Clone, Debug)]
struct Assignment<'a> {
    offset: Vec<u64>,
    shape: Vec<u64>,
    source: Option<Fragment<'a>>,
}

fn partition_assignments<'a>(
    logical_offset: &[u64],
    logical_shape: &[u64],
    sources: &[Fragment<'a>],
    budget: &mut PlanningBudget<'_>,
) -> Result<Vec<Assignment<'a>>> {
    let ndim = logical_shape.len();
    let mut boundaries: Vec<BTreeSet<u64>> = Vec::with_capacity(ndim);
    for dim in 0..ndim {
        let end = logical_offset[dim]
            .checked_add(logical_shape[dim])
            .ok_or_else(|| validation("logical box arithmetic overflow"))?;
        boundaries.push(BTreeSet::from([logical_offset[dim], end]));
    }

    let mut ordered_sources = sources.to_vec();
    ordered_sources.sort_by(source_cmp);
    let mut overlaps = Vec::new();
    for source in ordered_sources {
        if let Some((offset, shape)) = intersection(
            &source.tensor.logical_offset,
            &source.tensor.logical_shape,
            logical_offset,
            logical_shape,
        )? {
            for dim in 0..ndim {
                boundaries[dim].insert(offset[dim]);
                boundaries[dim].insert(
                    offset[dim]
                        .checked_add(shape[dim])
                        .ok_or_else(|| validation("logical box arithmetic overflow"))?,
                );
            }
            overlaps.push((source, offset, shape));
        }
    }

    let ordered_boundaries: Vec<Vec<u64>> = boundaries
        .into_iter()
        .map(|axis| axis.into_iter().collect())
        .collect();
    let intervals: Vec<Vec<(u64, u64)>> = ordered_boundaries
        .iter()
        .map(|axis| {
            axis.windows(2)
                .filter_map(|pair| {
                    let begin = pair[0];
                    let end = pair[1];
                    (begin < end).then_some((begin, end - begin))
                })
                .collect()
        })
        .collect();
    let cell_count = intervals.iter().try_fold(1_usize, |product, axis| {
        product
            .checked_mul(axis.len())
            .ok_or_else(|| validation("partition cell count overflow"))
    })?;
    budget.consume_partition_cells(cell_count)?;

    let boundary_indexes: Vec<BTreeMap<u64, usize>> = ordered_boundaries
        .iter()
        .map(|axis| {
            axis.iter()
                .copied()
                .enumerate()
                .map(|(index, boundary)| (boundary, index))
                .collect()
        })
        .collect();
    let mut assignments: BTreeMap<Vec<usize>, Fragment<'a>> = BTreeMap::new();
    for (source, offset, shape) in overlaps {
        let mut ranges = Vec::with_capacity(ndim);
        for dim in 0..ndim {
            let begin = *boundary_indexes[dim]
                .get(&offset[dim])
                .ok_or_else(|| validation("missing partition boundary"))?;
            let end_value = offset[dim]
                .checked_add(shape[dim])
                .ok_or_else(|| validation("logical box arithmetic overflow"))?;
            let end = *boundary_indexes[dim]
                .get(&end_value)
                .ok_or_else(|| validation("missing partition boundary"))?;
            ranges.push(begin..end);
        }
        visit_coordinates(&ranges, |coordinate| {
            budget.consume_candidate_assignments(1)?;
            assignments
                .entry(coordinate.to_vec())
                .or_insert_with(|| source.clone());
            Ok(())
        })?;
        if assignments.len() == cell_count {
            break;
        }
    }

    let ranges: Vec<_> = intervals.iter().map(|axis| 0..axis.len()).collect();
    let mut result = Vec::with_capacity(cell_count);
    visit_coordinates(&ranges, |coordinate| {
        let mut offset = Vec::with_capacity(ndim);
        let mut shape = Vec::with_capacity(ndim);
        for dim in 0..ndim {
            let (begin, extent) = intervals[dim][coordinate[dim]];
            offset.push(begin);
            shape.push(extent);
        }
        result.push(Assignment {
            offset,
            shape,
            source: assignments.get(coordinate).cloned(),
        });
        Ok(())
    })?;
    Ok(result)
}

fn fully_covers(
    logical_offset: &[u64],
    logical_shape: &[u64],
    fragments: &[Fragment<'_>],
    budget: &mut PlanningBudget<'_>,
) -> Result<bool> {
    if fragments.is_empty() {
        return Ok(false);
    }
    Ok(
        partition_assignments(logical_offset, logical_shape, fragments, budget)?
            .iter()
            .all(|assignment| assignment.source.is_some()),
    )
}

fn select_source_dp<'a>(
    global_shape: &[u64],
    sources: &[Fragment<'a>],
    budget: &mut PlanningBudget<'_>,
) -> Result<Vec<Fragment<'a>>> {
    let mut by_dp: BTreeMap<u64, Vec<Fragment<'a>>> = BTreeMap::new();
    for source in sources {
        by_dp
            .entry(source.tensor.rank.dp)
            .or_default()
            .push(source.clone());
    }
    let logical_offset = vec![0; global_shape.len()];
    for (_, mut candidates) in by_dp {
        candidates.sort_by(source_cmp);
        if fully_covers(&logical_offset, global_shape, &candidates, budget)? {
            return Ok(candidates);
        }
    }
    Err(validation(format!(
        "source coverage hole for tensor: {}",
        sources
            .first()
            .map(|fragment| fragment.tensor.tensor_id.as_str())
            .unwrap_or("<unknown>")
    )))
}

fn validate_target_coverage(
    global_shape: &[u64],
    targets: &[Fragment<'_>],
    budget: &mut PlanningBudget<'_>,
) -> Result<()> {
    let mut by_dp: BTreeMap<u64, Vec<Fragment<'_>>> = BTreeMap::new();
    for target in targets {
        by_dp
            .entry(target.tensor.rank.dp)
            .or_default()
            .push(target.clone());
    }
    let logical_offset = vec![0; global_shape.len()];
    for (dp, fragments) in by_dp {
        if !fully_covers(&logical_offset, global_shape, &fragments, budget)? {
            return Err(validation(format!(
                "target coverage hole for tensor {}, dp={dp}",
                targets
                    .first()
                    .map(|fragment| fragment.tensor.tensor_id.as_str())
                    .unwrap_or("<unknown>")
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct SourceIndexNode<'a> {
    logical_offset: Vec<u64>,
    logical_end: Vec<u64>,
    source: Option<Fragment<'a>>,
    left: Option<Box<SourceIndexNode<'a>>>,
    right: Option<Box<SourceIndexNode<'a>>>,
}

fn build_source_node<'a>(
    mut sources: Vec<Fragment<'a>>,
    ndim: usize,
) -> Result<SourceIndexNode<'a>> {
    if sources.len() == 1 {
        let source = sources.pop().expect("length checked");
        let logical_end = source
            .tensor
            .logical_offset
            .iter()
            .zip(&source.tensor.logical_shape)
            .map(|(offset, extent)| {
                offset
                    .checked_add(*extent)
                    .ok_or_else(|| validation("logical box arithmetic overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(SourceIndexNode {
            logical_offset: source.tensor.logical_offset.clone(),
            logical_end,
            source: Some(source),
            left: None,
            right: None,
        });
    }

    let split_axis = (0..ndim)
        .max_by_key(|dim| {
            let mut minimum = u128::MAX;
            let mut maximum = 0_u128;
            for source in &sources {
                let center = u128::from(source.tensor.logical_offset[*dim]) * 2
                    + u128::from(source.tensor.logical_shape[*dim]);
                minimum = minimum.min(center);
                maximum = maximum.max(center);
            }
            (maximum - minimum, std::cmp::Reverse(*dim))
        })
        .ok_or_else(|| validation("source spatial index rank must be positive"))?;
    sources.sort_by(|left, right| {
        let left_center = u128::from(left.tensor.logical_offset[split_axis]) * 2
            + u128::from(left.tensor.logical_shape[split_axis]);
        let right_center = u128::from(right.tensor.logical_offset[split_axis]) * 2
            + u128::from(right.tensor.logical_shape[split_axis]);
        left_center
            .cmp(&right_center)
            .then_with(|| source_cmp(left, right))
    });
    let right_sources = sources.split_off(sources.len() / 2);
    let left = Box::new(build_source_node(sources, ndim)?);
    let right = Box::new(build_source_node(right_sources, ndim)?);
    let logical_offset = (0..ndim)
        .map(|dim| left.logical_offset[dim].min(right.logical_offset[dim]))
        .collect();
    let logical_end = (0..ndim)
        .map(|dim| left.logical_end[dim].max(right.logical_end[dim]))
        .collect();
    Ok(SourceIndexNode {
        logical_offset,
        logical_end,
        source: None,
        left: Some(left),
        right: Some(right),
    })
}

#[derive(Debug)]
struct SourceSpatialIndex<'a> {
    root: SourceIndexNode<'a>,
    ndim: usize,
}

impl<'a> SourceSpatialIndex<'a> {
    fn build(mut sources: Vec<Fragment<'a>>, budget: &mut PlanningBudget<'_>) -> Result<Self> {
        sources.sort_by(source_cmp);
        if sources.is_empty() {
            return Err(validation(
                "source spatial index requires at least one fragment",
            ));
        }
        let ndim = sources[0].tensor.logical_shape.len();
        let nodes = sources
            .len()
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| validation("source spatial index size overflow"))?;
        budget.consume_candidate_assignments(nodes)?;
        Ok(Self {
            root: build_source_node(sources, ndim)?,
            ndim,
        })
    }

    fn candidates(
        &self,
        logical_offset: &[u64],
        logical_shape: &[u64],
        budget: &mut PlanningBudget<'_>,
    ) -> Result<Vec<Fragment<'a>>> {
        if logical_offset.len() != self.ndim || logical_shape.len() != self.ndim {
            return Err(validation("source spatial index query rank mismatch"));
        }
        let logical_end = logical_offset
            .iter()
            .zip(logical_shape)
            .map(|(offset, extent)| {
                offset
                    .checked_add(*extent)
                    .ok_or_else(|| validation("logical box arithmetic overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut result = Vec::new();

        fn visit<'a>(
            node: &SourceIndexNode<'a>,
            logical_offset: &[u64],
            logical_end: &[u64],
            budget: &mut PlanningBudget<'_>,
            result: &mut Vec<Fragment<'a>>,
        ) -> Result<()> {
            budget.consume_candidate_assignments(1)?;
            if (0..logical_offset.len()).any(|dim| {
                node.logical_end[dim] <= logical_offset[dim]
                    || logical_end[dim] <= node.logical_offset[dim]
            }) {
                return Ok(());
            }
            if let Some(source) = &node.source {
                result.push(source.clone());
                return Ok(());
            }
            let left = node
                .left
                .as_deref()
                .ok_or_else(|| validation("source spatial index node is incomplete"))?;
            let right = node
                .right
                .as_deref()
                .ok_or_else(|| validation("source spatial index node is incomplete"))?;
            visit(left, logical_offset, logical_end, budget, result)?;
            visit(right, logical_offset, logical_end, budget, result)?;
            Ok(())
        }

        visit(
            &self.root,
            logical_offset,
            &logical_end,
            budget,
            &mut result,
        )?;
        result.sort_by(source_cmp);
        Ok(result)
    }
}

#[derive(Debug)]
struct CopyOperation {
    index: usize,
    tensor_id: String,
    source_rank: ParallelRank,
    target_rank: ParallelRank,
    source_dev: u64,
    target_dev: u64,
    source_address: u64,
    target_address: u64,
    inner_bytes: u64,
    outer_loop_counts: Vec<u64>,
    source_strides_bytes: Vec<u64>,
    target_strides_bytes: Vec<u64>,
}

impl CopyOperation {
    fn for_each_segment<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(usize, u64, u64, u64) -> Result<()>,
    {
        let ranges: Vec<_> = self
            .outer_loop_counts
            .iter()
            .map(|count| {
                usize::try_from(*count)
                    .map(|count| 0..count)
                    .map_err(|_| validation("outer loop count exceeds usize"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut segment_index = 0_usize;
        visit_coordinates(&ranges, |coordinate| {
            let mut source = u128::from(self.source_address);
            let mut target = u128::from(self.target_address);
            for ((index, source_stride), target_stride) in coordinate
                .iter()
                .zip(&self.source_strides_bytes)
                .zip(&self.target_strides_bytes)
            {
                source += (*index as u128) * u128::from(*source_stride);
                target += (*index as u128) * u128::from(*target_stride);
            }
            let source = u64::try_from(source)
                .map_err(|_| validation("source segment address exceeds uint64"))?;
            let target = u64::try_from(target)
                .map_err(|_| validation("target segment address exceeds uint64"))?;
            visitor(segment_index, source, target, self.inner_bytes)?;
            segment_index += 1;
            Ok(())
        })
    }
}

fn copy_geometry(
    source: &RuntimeTensorManifest,
    target: &RuntimeTensorManifest,
    offset: &[u64],
    shape: &[u64],
) -> Result<(u64, u64, u64, Vec<u64>, Vec<u64>, Vec<u64>)> {
    let address = |tensor: &RuntimeTensorManifest| -> Result<u64> {
        let mut result = u128::from(tensor.addr);
        for ((begin, fragment_begin), stride) in offset
            .iter()
            .zip(&tensor.logical_offset)
            .zip(&tensor.strides_bytes)
        {
            let relative = begin
                .checked_sub(*fragment_begin)
                .ok_or_else(|| validation("copy begins before fragment"))?;
            result = result
                .checked_add(u128::from(relative) * u128::from(*stride))
                .ok_or_else(|| validation("copy address arithmetic overflow"))?;
        }
        u64::try_from(result).map_err(|_| validation("copy address exceeds uint64"))
    };

    let source_address = address(source)?;
    let target_address = address(target)?;
    let mut suffix_begin = shape.len() - 1;
    let mut inner_bytes = shape[shape.len() - 1]
        .checked_mul(target.itemsize)
        .ok_or_else(|| validation("copy inner byte count overflow"))?;
    for dim in (0..shape.len() - 1).rev() {
        if source.strides_bytes[dim] != inner_bytes || target.strides_bytes[dim] != inner_bytes {
            break;
        }
        inner_bytes = inner_bytes
            .checked_mul(shape[dim])
            .ok_or_else(|| validation("copy inner byte count overflow"))?;
        suffix_begin = dim;
    }
    Ok((
        source_address,
        target_address,
        inner_bytes,
        shape[..suffix_begin].to_vec(),
        source.strides_bytes[..suffix_begin].to_vec(),
        target.strides_bytes[..suffix_begin].to_vec(),
    ))
}

fn validate_tensor_compatibility(
    tensor_id: &str,
    source: &RuntimeTensorManifest,
    target: &RuntimeTensorManifest,
) -> Result<()> {
    if source.semantic != target.semantic {
        return Err(validation(format!(
            "semantic fingerprint mismatch: {tensor_id}"
        )));
    }
    if source.global_shape != target.global_shape {
        return Err(validation(format!("global shape mismatch: {tensor_id}")));
    }
    if source.dtype != target.dtype || source.itemsize != target.itemsize {
        return Err(validation(format!(
            "dtype or itemsize mismatch: {tensor_id}"
        )));
    }
    Ok(())
}

pub(crate) fn plan_scr_transfer(
    request: &ConversionRequest,
    max_chunk_bytes: Option<usize>,
    limits: &ConversionLimits,
) -> Result<ScrTransferPlan> {
    request.validate_with_limits(limits)?;
    let chunk_bytes = max_chunk_bytes.unwrap_or(limits.max_staging_chunk_bytes);
    if chunk_bytes == 0 || chunk_bytes > limits.max_staging_chunk_bytes {
        return Err(validation(format!(
            "max_chunk_bytes must be between 1 and max_staging_chunk_bytes={}",
            limits.max_staging_chunk_bytes
        )));
    }

    let source_groups = group_fragments("source", &request.source_manifest.tensors)?;
    let target_groups = group_fragments("target", &request.target_manifest.tensors)?;
    if source_groups.is_empty() || source_groups.keys().ne(target_groups.keys()) {
        return Err(validation("source and target tensor sets mismatch"));
    }

    let mut budget = PlanningBudget::new(limits);
    let mut operations = Vec::new();
    for (tensor_id, targets) in &target_groups {
        let sources = source_groups
            .get(tensor_id)
            .ok_or_else(|| validation("source and target tensor sets mismatch"))?;
        let source_first = &sources[0].tensor;
        let target_first = &targets[0].tensor;
        validate_tensor_compatibility(tensor_id, source_first, target_first)?;
        validate_target_coverage(&target_first.global_shape, targets, &mut budget)?;
        let eligible_sources = select_source_dp(&source_first.global_shape, sources, &mut budget)?;
        let source_index = SourceSpatialIndex::build(eligible_sources, &mut budget)?;

        let mut ordered_targets = targets.clone();
        ordered_targets.sort_by(|left, right| left.id.cmp(&right.id));
        for target in ordered_targets {
            let candidates = source_index.candidates(
                &target.tensor.logical_offset,
                &target.tensor.logical_shape,
                &mut budget,
            )?;
            let assignments = partition_assignments(
                &target.tensor.logical_offset,
                &target.tensor.logical_shape,
                &candidates,
                &mut budget,
            )?;
            for assignment in assignments {
                let source = assignment.source.ok_or_else(|| {
                    validation(format!(
                        "source coverage hole for target fragment: {}",
                        target.id
                    ))
                })?;
                let (
                    source_address,
                    target_address,
                    inner_bytes,
                    outer_loop_counts,
                    source_strides_bytes,
                    target_strides_bytes,
                ) = copy_geometry(
                    source.tensor,
                    target.tensor,
                    &assignment.offset,
                    &assignment.shape,
                )?;
                operations.push(CopyOperation {
                    index: operations.len(),
                    tensor_id: tensor_id.clone(),
                    source_rank: source.tensor.rank,
                    target_rank: target.tensor.rank,
                    source_dev: source.tensor.dev,
                    target_dev: target.tensor.dev,
                    source_address,
                    target_address,
                    inner_bytes,
                    outer_loop_counts,
                    source_strides_bytes,
                    target_strides_bytes,
                });
                if operations.len() > limits.max_copy_operations {
                    return Err(validation(format!(
                        "copy operation count exceeds max_copy_operations={}",
                        limits.max_copy_operations
                    )));
                }
            }
        }
    }

    let mut transfers = Vec::new();
    for operation in operations {
        operation.for_each_segment(|segment_index, source, target, nbytes| {
            let mut offset = 0_u64;
            let mut chunk_index = 0_usize;
            let chunk_bytes = chunk_bytes as u64;
            while offset < nbytes {
                let chunk_nbytes = chunk_bytes.min(nbytes - offset);
                let key = format!(
                    "wt/tx-{}-copy{:08}-{}-{}/S-p{}t{}e{}/T-p{}t{}e{}",
                    request.plan_id,
                    operation.index,
                    segment_index,
                    chunk_index,
                    operation.source_rank.pp,
                    operation.source_rank.tp,
                    operation.source_rank.ep,
                    operation.target_rank.pp,
                    operation.target_rank.tp,
                    operation.target_rank.ep,
                );
                if transfers.len() >= limits.max_transfer_tasks {
                    return Err(validation(format!(
                        "transfer task count exceeds max_transfer_tasks={}",
                        limits.max_transfer_tasks
                    )));
                }
                if key.len() > limits.max_key_bytes {
                    return Err(validation(format!(
                        "transfer key exceeds max_key_bytes={}",
                        limits.max_key_bytes
                    )));
                }
                transfers.push(ScrTransferTask {
                    key,
                    tensor_id: operation.tensor_id.clone(),
                    source: DeviceMemoryLocation {
                        dev: operation.source_dev,
                        addr: source
                            .checked_add(offset)
                            .ok_or_else(|| validation("source chunk address overflow"))?,
                    },
                    target: DeviceMemoryLocation {
                        dev: operation.target_dev,
                        addr: target
                            .checked_add(offset)
                            .ok_or_else(|| validation("target chunk address overflow"))?,
                    },
                    nbytes: chunk_nbytes,
                });
                offset += chunk_nbytes;
                chunk_index += 1;
            }
            Ok(())
        })?;
    }
    ScrTransferPlan::new(request.plan_id.clone(), transfers, limits)
}
