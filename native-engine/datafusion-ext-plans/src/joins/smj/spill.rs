// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Bounded replay of an oversized, non-null equality group.

use std::{
    io::{Read, Write},
    sync::Arc,
};

use arrow::{
    array::{ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array},
    datatypes::SchemaRef,
};
use auron_memmgr::{
    metrics::SpillMetrics,
    spill::{Spill, SpillCompressedReader, try_new_file_spill},
};
use datafusion::common::Result;
use datafusion_ext_commons::{
    arrow::{array_size::BatchSize, selection::take_cols},
    df_execution_err,
    io::read_one_batch,
};

use crate::{
    common::execution_context::{WrappedRecordBatchSender, WrappedSenderTrait},
    joins::{IdxRange, JoinParams, join_filter::RunBatch, join_utils::JoinType, product_blocks},
};

pub(super) struct SpilledGroup {
    pub(super) data: Box<dyn Spill>,
    schema: SchemaRef,
    pub(super) num_rows: usize,
}

impl SpilledGroup {
    pub(super) fn new(schema: SchemaRef, metrics: &SpillMetrics) -> Result<Self> {
        let data = try_new_file_spill(metrics)?;
        metrics.mem_spill_count.add(1);
        Ok(Self {
            data,
            schema,
            num_rows: 0,
        })
    }

    fn reader(&mut self) -> GroupReader<'_> {
        // The exclusive borrow prevents simultaneous readers: FileSpill
        // clones share an OS file position.
        GroupReader {
            reader: self.data.get_compressed_reader(),
            schema: self.schema.clone(),
            remaining: self.num_rows,
        }
    }
}

struct GroupReader<'a> {
    reader: SpillCompressedReader<'a>,
    schema: SchemaRef,
    remaining: usize,
}

impl GroupReader<'_> {
    fn next(&mut self) -> Result<Option<RecordBatch>> {
        let Some((rows, cols)) = read_one_batch(&mut self.reader, &self.schema)? else {
            if self.remaining != 0 {
                return df_execution_err!(
                    "SMJ group spill ended with {} rows missing",
                    self.remaining
                );
            }
            return Ok(None);
        };
        if rows == 0 || rows > self.remaining {
            return df_execution_err!("SMJ group spill has an invalid row count: {rows}");
        }
        self.remaining -= rows;
        Ok(Some(RecordBatch::try_new_with_options(
            self.schema.clone(),
            cols,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )?))
    }
}

/// Stream one side in bounded chunks and replay the other. Only full outer
/// with a residual predicate needs match state for the replayed side; stream
/// that state through fresh files on each pass instead of retaining a bitmap
/// for the whole group. Predicates are evaluated once per visited candidate.
pub(super) async fn join_spilled_group(
    mut left: SpilledGroup,
    mut right: SpilledGroup,
    params: &JoinParams,
    sender: &Arc<WrappedRecordBatchSender>,
) -> Result<usize> {
    use JoinType::*;
    let primary_right = matches!(params.join_type, Right | RightSemi | RightAnti);
    let (primary, secondary) = if primary_right {
        (&mut right, &mut left)
    } else {
        (&mut left, &mut right)
    };
    let emits_pairs = matches!(params.join_type, Inner | Left | Right | Full);
    let track_secondary = params.join_type == Full && params.residual_filter.is_some();
    let metrics = sender.exec_ctx().spill_metrics();
    let mut secondary_matches: Option<Box<dyn Spill>> = None;
    let mut output_rows = 0;
    let null_primary = RecordBatch::new_empty(primary.schema.clone());
    let null_secondary = RecordBatch::new_empty(secondary.schema.clone());
    let mut primary_reader = primary.reader();

    while let Some(own) = primary_reader.next()? {
        let mut matched = vec![false; own.num_rows()];
        if !emits_pairs && params.residual_filter.is_none() {
            // Both groups are nonempty. Every own row matches without
            // reading or expanding the opposite group.
            matched.fill(true);
        } else {
            let mut other_reader = secondary.reader();
            let mut old_matches = secondary_matches
                .as_ref()
                .map(|s| s.get_compressed_reader());
            let mut new_matches = if track_secondary {
                Some(try_new_file_spill(metrics)?)
            } else {
                None
            };
            let mut new_writer = new_matches.as_mut().map(|s| s.get_compressed_writer());
            while let Some(other) = other_reader.next()? {
                let mut other_matched = if track_secondary {
                    vec![0u8; other.num_rows()]
                } else {
                    vec![]
                };
                if let Some(reader) = &mut old_matches {
                    reader.read_exact(&mut other_matched)?;
                }
                let (lbatch, rbatch) = if primary_right {
                    (&other, &own)
                } else {
                    (&own, &other)
                };
                let lranges = [IdxRange {
                    batch_idx: 0,
                    start: 0,
                    end: lbatch.num_rows(),
                }];
                let rranges = [IdxRange {
                    batch_idx: 0,
                    start: 0,
                    end: rbatch.num_rows(),
                }];
                let mut pending = RunBatch::default();
                let bytes_per_pair = (lbatch.get_batch_mem_size() / lbatch.num_rows())
                    .saturating_add(rbatch.get_batch_mem_size() / rbatch.num_rows())
                    .max(1);
                let max_pairs = params
                    .batch_size
                    .max(1)
                    .min((1024 * 1024 / bytes_per_pair).max(1));
                for block in product_blocks(&lranges, &rranges, max_pairs) {
                    let own_range = if primary_right {
                        block.rrange
                    } else {
                        block.lrange
                    };
                    if !emits_pairs && matched[own_range.start..own_range.end].iter().all(|&m| m) {
                        continue;
                    }
                    let (mut lpos, mut rpos) = (vec![], vec![]);
                    if let Some(filter) = &params.residual_filter {
                        pending.push(&block, 0, 0);
                        pending.evaluate(
                            filter,
                            std::slice::from_ref(lbatch),
                            std::slice::from_ref(rbatch),
                            |run| {
                                lpos.extend(run.lpos.iter().map(|&p| p.wrapping_add(run.lrow)));
                                rpos.extend(run.rpos.iter().map(|&p| p.wrapping_add(run.rrow)));
                            },
                        )?;
                    } else {
                        for l in block.lrange.start..block.lrange.end {
                            for r in block.rrange.start..block.rrange.end {
                                lpos.push(l as u32);
                                rpos.push(r as u32);
                            }
                        }
                    }
                    let (own_pos, other_pos) = if primary_right {
                        (&rpos, &lpos)
                    } else {
                        (&lpos, &rpos)
                    };
                    for &p in own_pos {
                        matched[p as usize] = true;
                    }
                    if track_secondary {
                        for &p in other_pos {
                            other_matched[p as usize] = 1;
                        }
                    }
                    if emits_pairs && !lpos.is_empty() {
                        let lpos = lpos.into_iter().map(Some).collect::<Vec<_>>();
                        let rpos = rpos.into_iter().map(Some).collect::<Vec<_>>();
                        output_rows +=
                            emit(params, sender, lbatch, rbatch, lpos, rpos, None).await?;
                    }
                    tokio::task::yield_now().await;
                }
                if let Some(writer) = &mut new_writer {
                    writer.write_all(&other_matched)?;
                }
                if !emits_pairs && matched.iter().all(|&m| m) {
                    break;
                }
                tokio::task::yield_now().await;
            }
            if let Some(writer) = new_writer.take() {
                writer.finish()?;
            }
            drop(new_writer);
            drop(old_matches);
            drop(other_reader);
            if track_secondary {
                secondary_matches = new_matches;
            }
        }

        let selected = matched
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| {
                let keep = match params.join_type {
                    Left | Right | Full | LeftAnti | RightAnti => !m,
                    LeftSemi | RightSemi => m,
                    Existence => true,
                    Inner => false,
                };
                keep.then_some(Some(i as u32))
            })
            .collect::<Vec<_>>();
        if !selected.is_empty() {
            let exists = (params.join_type == Existence).then(|| matched.clone());
            let nulls = vec![None; selected.len()];
            let (lbatch, rbatch, lpos, rpos) = if primary_right {
                (&null_secondary, &own, nulls, selected)
            } else {
                (&own, &null_secondary, selected, nulls)
            };
            output_rows += emit(params, sender, lbatch, rbatch, lpos, rpos, exists).await?;
        }
        tokio::task::yield_now().await;
    }
    drop(primary_reader);

    if let Some(flags) = secondary_matches {
        let mut data_reader = secondary.reader();
        let mut flag_reader = flags.get_compressed_reader();
        while let Some(other) = data_reader.next()? {
            let mut matched = vec![0; other.num_rows()];
            flag_reader.read_exact(&mut matched)?;
            let selected = matched
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| (m == 0).then_some(Some(i as u32)))
                .collect::<Vec<_>>();
            if !selected.is_empty() {
                let nulls = vec![None; selected.len()];
                output_rows +=
                    emit(params, sender, &null_primary, &other, nulls, selected, None).await?;
            }
            tokio::task::yield_now().await;
        }
    }
    Ok(output_rows)
}

async fn emit(
    params: &JoinParams,
    sender: &Arc<WrappedRecordBatchSender>,
    left: &RecordBatch,
    right: &RecordBatch,
    lpos: Vec<Option<u32>>,
    rpos: Vec<Option<u32>>,
    exists: Option<Vec<bool>>,
) -> Result<usize> {
    let timer = params.output_time.timer();
    let rows = lpos.len();
    let projection = &params.projection;
    let mut cols = take_cols(
        &projection.output_left(left.columns()),
        UInt32Array::from(lpos),
    )?;
    cols.extend(take_cols(
        &projection.output_right(right.columns()),
        UInt32Array::from(rpos),
    )?);
    if let Some(pos) = projection.existence_output {
        cols.insert(
            pos,
            Arc::new(BooleanArray::from(exists.expect("existence join flags"))) as ArrayRef,
        );
    }
    let batch = RecordBatch::try_new_with_options(
        projection.schema.clone(),
        cols,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )?;
    drop(timer);
    sender.send(batch).await;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::Int32Array,
        compute::SortOptions,
        datatypes::{DataType, Field, Schema},
        util::display::array_value_to_string,
    };
    use datafusion::{
        common::ScalarValue,
        execution::TaskContext,
        logical_expr::Operator,
        physical_expr::{
            PhysicalExprRef,
            expressions::{BinaryExpr, Column, Literal},
        },
        physical_plan::{
            joins::utils::build_join_schema,
            metrics::{ExecutionPlanMetricsSet, Time},
        },
    };

    use super::*;
    use crate::{
        common::execution_context::ExecutionContext,
        cur_forward,
        joins::{
            JoinProjection,
            join_filter::JoinFilter,
            smj::{
                group_merge::{GroupAction, GroupMerger},
                tests::cursor,
            },
        },
    };

    fn write_group(batch: &RecordBatch, metrics: &SpillMetrics) -> Result<SpilledGroup> {
        let mut group = SpilledGroup::new(batch.schema(), metrics)?;
        let mut writer = group.data.get_compressed_writer();
        for row in 0..batch.num_rows() {
            let part = batch.slice(row, 1);
            datafusion_ext_commons::io::write_one_batch(1, part.columns(), &mut writer)?;
        }
        writer.finish()?;
        group.num_rows = batch.num_rows();
        Ok(group)
    }

    #[test]
    fn truncated_group_does_not_silently_drop_rows() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])?;
        let metrics = SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let mut group = write_group(&batch, &metrics)?;
        group.num_rows = 2;
        let mut reader = group.reader();
        assert_eq!(reader.next()?.expect("first batch").num_rows(), 1);
        let error = reader.next().expect_err("missing batch must fail");
        assert!(error.to_string().contains("1 rows missing"));
        Ok(())
    }

    #[tokio::test]
    async fn cancelled_replay_drops_spill_files() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1; 64]))],
        )?;
        let output_schema = Arc::new(Schema::new(vec![
            schema.field(0).clone(),
            schema.field(0).clone(),
        ]));
        let ctx = ExecutionContext::new(
            Arc::new(TaskContext::default()),
            0,
            output_schema.clone(),
            &ExecutionPlanMetricsSet::new(),
        );
        let left = write_group(&batch, ctx.spill_metrics())?;
        let right = write_group(&batch, ctx.spill_metrics())?;
        let mut bytes = vec![];
        left.data.get_buf_reader().read_to_end(&mut bytes)?;
        right.data.get_buf_reader().read_to_end(&mut bytes)?;
        let expected_bytes = bytes.len();
        let params = JoinParams {
            join_type: JoinType::Inner,
            left_schema: schema.clone(),
            right_schema: schema.clone(),
            output_schema: output_schema.clone(),
            left_keys: vec![Arc::new(Column::new("k", 0))],
            right_keys: vec![Arc::new(Column::new("k", 0))],
            key_data_types: vec![DataType::Int32],
            sort_options: vec![SortOptions::default()],
            projection: JoinProjection::try_new(
                JoinType::Inner,
                &output_schema,
                &schema,
                &schema,
                &[0, 1],
            )?,
            join_filter: None,
            residual_filter: None,
            batch_size: 1,
            is_null_aware_anti_join: false,
            output_time: Time::default(),
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let sender = WrappedRecordBatchSender::new(ctx.clone(), tx);
        let task =
            tokio::spawn(async move { join_spilled_group(left, right, &params, &sender).await });
        rx.recv()
            .await
            .expect("first output")
            .expect("successful output");
        task.abort();
        assert!(task.await.expect_err("replay cancelled").is_cancelled());
        assert_eq!(ctx.spill_metrics().disk_spill_size.value(), expected_bytes);
        Ok(())
    }

    #[tokio::test]
    async fn spilled_group_all_join_types_filters_and_projections() -> Result<()> {
        use JoinType::*;
        let lvalues = vec![None, Some(1), Some(7), Some(20), Some(30)];
        let rvalues = vec![None, Some(5), Some(15), Some(35)];
        let input = |prefix: &str, values: &[Option<i32>]| {
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new(format!("{prefix}k"), DataType::Int32, true),
                    Field::new(format!("{prefix}v"), DataType::Int32, true),
                ])),
                vec![
                    Arc::new(Int32Array::from(vec![1; values.len()])) as ArrayRef,
                    Arc::new(Int32Array::from(values.to_vec())),
                ],
            )
            .expect("valid test input")
        };
        let left = input("l", &lvalues);
        let right = input("r", &rvalues);
        for jt in [
            Inner, Left, Right, Full, LeftSemi, RightSemi, LeftAnti, RightAnti, Existence,
        ] {
            for mode in 0..5 {
                let mut pairs = vec![];
                let mut lm = vec![false; lvalues.len()];
                let mut rm = vec![false; rvalues.len()];
                for (l, lv) in lvalues.iter().enumerate() {
                    for (r, rv) in rvalues.iter().enumerate() {
                        let matched = match mode {
                            0 | 1 => true,
                            2 | 3 => false,
                            _ => lv.zip(*rv).is_some_and(|(l, r)| l > r),
                        };
                        if matched {
                            pairs.push((Some(l), Some(r)));
                            lm[l] = true;
                            rm[r] = true;
                        }
                    }
                }
                let key = |v: Option<i32>| v.map_or_else(String::new, |v| v.to_string());
                let expected = match jt {
                    Inner | Left | Right | Full => {
                        if matches!(jt, Left | Full) {
                            pairs.extend(
                                lm.iter()
                                    .enumerate()
                                    .filter(|(_, m)| !**m)
                                    .map(|(l, _)| (Some(l), None)),
                            );
                        }
                        if matches!(jt, Right | Full) {
                            pairs.extend(
                                rm.iter()
                                    .enumerate()
                                    .filter(|(_, m)| !**m)
                                    .map(|(r, _)| (None, Some(r))),
                            );
                        }
                        pairs
                            .iter()
                            .map(|&(l, r)| {
                                vec![
                                    key(l.and_then(|l| lvalues[l])),
                                    key(r.and_then(|r| rvalues[r])),
                                ]
                            })
                            .collect::<Vec<_>>()
                    }
                    LeftSemi | LeftAnti => lm
                        .iter()
                        .enumerate()
                        .filter(|(_, m)| **m == (jt == LeftSemi))
                        .map(|(l, _)| vec![key(lvalues[l])])
                        .collect(),
                    RightSemi | RightAnti => rm
                        .iter()
                        .enumerate()
                        .filter(|(_, m)| **m == (jt == RightSemi))
                        .map(|(r, _)| vec![key(rvalues[r])])
                        .collect(),
                    Existence => lm
                        .iter()
                        .enumerate()
                        .map(|(l, m)| vec![key(lvalues[l]), m.to_string()])
                        .collect(),
                };
                for empty_projection in [false, true] {
                    let schema = if jt == Existence {
                        Arc::new(Schema::new(
                            [
                                left.schema().fields().to_vec(),
                                vec![Arc::new(Field::new("exists", DataType::Boolean, false))],
                            ]
                            .concat(),
                        ))
                    } else {
                        Arc::new(
                            build_join_schema(&left.schema(), &right.schema(), &jt.try_into()?).0,
                        )
                    };
                    let projected = if empty_projection {
                        vec![]
                    } else {
                        match jt {
                            Inner | Left | Right | Full => vec![1, 3],
                            Existence => vec![1, 2],
                            _ => vec![1],
                        }
                    };
                    let expr: Option<PhysicalExprRef> = match mode {
                        0 => None,
                        1 => Some(Arc::new(Literal::new(ScalarValue::Boolean(Some(true))))),
                        2 => Some(Arc::new(Literal::new(ScalarValue::Boolean(Some(false))))),
                        3 => Some(Arc::new(Literal::new(ScalarValue::Boolean(None)))),
                        _ => Some(Arc::new(BinaryExpr::new(
                            Arc::new(Column::new("lv", 1)),
                            Operator::Gt,
                            Arc::new(Column::new("rv", 3)),
                        ))),
                    };
                    let projection = JoinProjection::try_new_with_filter(
                        jt,
                        &schema,
                        &left.schema(),
                        &right.schema(),
                        &projected,
                        expr.as_ref(),
                    )?;
                    let filter = expr
                        .map(|expr| {
                            JoinFilter::try_new_remapped(
                                expr,
                                &left.schema(),
                                &right.schema(),
                                &projection.left_read,
                                &projection.right_read,
                                Time::default(),
                            )
                        })
                        .transpose()?
                        .map(Arc::new);
                    let params = JoinParams {
                        join_type: jt,
                        left_schema: left.schema(),
                        right_schema: right.schema(),
                        output_schema: schema,
                        left_keys: vec![Arc::new(Column::new("lk", 0))],
                        right_keys: vec![Arc::new(Column::new("rk", 0))],
                        key_data_types: vec![DataType::Int32],
                        sort_options: vec![SortOptions::default()],
                        projection,
                        join_filter: None,
                        residual_filter: filter,
                        batch_size: 2,
                        is_null_aware_anti_join: false,
                        output_time: Time::default(),
                    };
                    let (mut lc, mut rc) = (
                        cursor(left.clone(), 2, &params.projection.left_read),
                        cursor(right.clone(), 2, &params.projection.right_read),
                    );
                    cur_forward!(lc);
                    cur_forward!(rc);
                    let mut merger = GroupMerger::with_memory_limit(1);
                    assert_eq!(
                        merger.next(&mut lc, &mut rc).await?,
                        Some(GroupAction::SpillRequired)
                    );
                    let metrics = ExecutionPlanMetricsSet::new();
                    let ctx = ExecutionContext::new(
                        Arc::new(TaskContext::default()),
                        0,
                        params.projection.schema.clone(),
                        &metrics,
                    );
                    let (tx, mut rx) = tokio::sync::mpsc::channel(128);
                    let sender = WrappedRecordBatchSender::new(ctx.clone(), tx);
                    let (ls, mut rs) = merger
                        .spill_group(&mut lc, &mut rc, 2, ctx.spill_metrics())
                        .await?;
                    assert_eq!((ls.num_rows, rs.num_rows), (5, 4));
                    assert_eq!(
                        (lc.num_buffered_batches(), rc.num_buffered_batches()),
                        (0, 0)
                    );
                    // Two sequential readers must each see all chunks, before the join replays them
                    // again.
                    for _ in 0..2 {
                        let mut reader = rs.reader();
                        let mut rows = 0;
                        while let Some(batch) = reader.next()? {
                            rows += batch.num_rows();
                        }
                        assert_eq!(rows, 4);
                    }
                    let rows = join_spilled_group(ls, rs, &params, &sender).await?;
                    drop(sender);
                    let mut actual = vec![];
                    while let Some(batch) = rx.recv().await {
                        let batch = batch?;
                        assert!(batch.num_rows() <= 2);
                        for row in 0..batch.num_rows() {
                            actual.push(
                                batch
                                    .columns()
                                    .iter()
                                    .map(|col| array_value_to_string(col, row))
                                    .collect::<std::result::Result<Vec<_>, _>>()?,
                            );
                        }
                    }
                    let mut expected = if empty_projection {
                        vec![vec![]; expected.len()]
                    } else {
                        expected.clone()
                    };
                    actual.sort();
                    expected.sort();
                    assert_eq!(
                        actual, expected,
                        "{jt:?}, mode={mode}, empty={empty_projection}"
                    );
                    assert_eq!(rows, actual.len());
                    assert!(ctx.spill_metrics().disk_spill_size.value() > 0);
                    assert_eq!(merger.next(&mut lc, &mut rc).await?, None);
                }
            }
        }
        Ok(())
    }
}
