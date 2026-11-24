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

use std::{fmt::Debug, sync::Arc};

use arrow::{
    array::{Array, ArrayRef, AsArray, ListArray, RecordBatch, RecordBatchOptions},
    datatypes::{DataType, Schema, SchemaRef},
};
use arrow_schema::Field;
use datafusion::{
    common::{
        ColumnStatistics, Result, ScalarValue,
        stats::Precision,
        tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    },
    datasource::schema_adapter::{SchemaAdapter, SchemaAdapterFactory, SchemaMapper},
    parquet::arrow::PARQUET_FIELD_ID_META_KEY,
    physical_expr::{
        PhysicalExprRef,
        expressions::{CastExpr, Column, Literal},
        utils::{conjunction, split_conjunction},
    },
    physical_expr_adapter::{PhysicalExprAdapter, PhysicalExprAdapterFactory},
};
use datafusion_ext_commons::{
    arrow::cast::{cast_scan_input_array, parquet_names_eq_ignore_case},
    df_execution_err,
};

type CastColumn = dyn Fn(&ArrayRef, &Field) -> Result<ArrayRef> + Send + Sync;

/// Replacement for datafusion's removed `SchemaMapping`, re-implementing the
/// same field-remapping + per-column casting behavior that was dropped in
/// favor of `PhysicalExprAdapterFactory`.
pub struct AuronSchemaMapping {
    table_schema: SchemaRef,
    field_mappings: Vec<Option<usize>>,
    cast_column: Arc<CastColumn>,
}

impl Debug for AuronSchemaMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuronSchemaMapping")
            .field("table_schema", &self.table_schema)
            .field("field_mappings", &self.field_mappings)
            .finish()
    }
}

impl AuronSchemaMapping {
    pub fn new(
        table_schema: SchemaRef,
        field_mappings: Vec<Option<usize>>,
        cast_column: Arc<CastColumn>,
    ) -> Self {
        Self {
            table_schema,
            field_mappings,
            cast_column,
        }
    }
}

impl SchemaMapper for AuronSchemaMapping {
    fn map_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let cols = self
            .table_schema
            .fields()
            .iter()
            .zip(&self.field_mappings)
            .map(|(field, file_idx)| match file_idx {
                Some(batch_idx) => (self.cast_column)(batch.column(*batch_idx), field),
                None => Ok(arrow::array::new_null_array(
                    field.data_type(),
                    batch.num_rows(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RecordBatch::try_new_with_options(
            self.table_schema.clone(),
            cols,
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )?)
    }

    fn map_column_statistics(
        &self,
        file_col_statistics: &[ColumnStatistics],
    ) -> Result<Vec<ColumnStatistics>> {
        Ok(self
            .field_mappings
            .iter()
            .map(|file_idx| {
                file_idx
                    .and_then(|idx| file_col_statistics.get(idx))
                    .cloned()
                    .unwrap_or_else(|| ColumnStatistics {
                        null_count: Precision::Absent,
                        max_value: Precision::Absent,
                        min_value: Precision::Absent,
                        sum_value: Precision::Absent,
                        distinct_count: Precision::Absent,
                        byte_size: Precision::Absent,
                    })
            })
            .collect())
    }
}

pub mod internal_file_reader;
pub mod parallel;

#[derive(Debug)]
pub struct AuronSchemaAdapterFactory(pub Option<bool>);

impl SchemaAdapterFactory for AuronSchemaAdapterFactory {
    fn create(
        &self,
        projected_table_schema: SchemaRef,
        _table_schema: SchemaRef,
    ) -> Box<dyn SchemaAdapter> {
        Box::new(AuronSchemaAdapter::new(projected_table_schema, self.0))
    }
}

pub struct AuronSchemaAdapter {
    table_schema: SchemaRef,
    case_sensitive: Option<bool>,
}

impl AuronSchemaAdapter {
    pub fn new(table_schema: SchemaRef, case_sensitive: Option<bool>) -> Self {
        Self {
            table_schema,
            case_sensitive,
        }
    }

    fn resolve_file_column(&self, index: usize, file_schema: &Schema) -> Result<Option<usize>> {
        let field = self.table_schema.field(index);
        let mut matches = file_schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, file_field)| fields_match(field, file_field, self.case_sensitive));
        let first = matches.next().map(|(index, _)| index);
        if self.case_sensitive == Some(false) && matches.next().is_some() {
            return df_execution_err!(
                "Ambiguous Parquet field in case-insensitive mode: {}",
                field.name()
            );
        }
        Ok(first)
    }
}

impl SchemaAdapter for AuronSchemaAdapter {
    fn map_column_index(&self, index: usize, file_schema: &Schema) -> Option<usize> {
        self.resolve_file_column(index, file_schema).ok().flatten()
    }

    fn nested_field_names_ignore_case(&self) -> bool {
        self.case_sensitive == Some(false)
    }

    fn map_schema(&self, file_schema: &Schema) -> Result<(Arc<dyn SchemaMapper>, Vec<usize>)> {
        let mut projection = Vec::with_capacity(file_schema.fields().len());
        let mut field_mappings = vec![None; self.table_schema.fields().len()];

        for (file_idx, file_field) in file_schema.fields.iter().enumerate() {
            if let Some(table_idx) =
                self.table_schema.fields().iter().position(|table_field| {
                    fields_match(table_field, file_field, self.case_sensitive)
                })
            {
                if self.case_sensitive == Some(false) && field_mappings[table_idx].is_some() {
                    return df_execution_err!(
                        "Ambiguous Parquet field in case-insensitive mode: {}",
                        self.table_schema.field(table_idx).name()
                    );
                }
                field_mappings[table_idx] = Some(projection.len());
                projection.push(file_idx);
            }
        }

        let case_sensitive = self.case_sensitive;
        let schema_mapper: Arc<dyn SchemaMapper> = Arc::new(AuronSchemaMapping::new(
            self.table_schema.clone(),
            field_mappings,
            Arc::new(move |col, field| schema_adapter_cast_column(col, field, case_sensitive)),
        ));
        Ok((schema_mapper, projection))
    }
}

fn fields_match(table_field: &Field, file_field: &Field, case_sensitive: Option<bool>) -> bool {
    if let (Some(table_id), Some(file_id)) = (
        table_field.metadata().get(PARQUET_FIELD_ID_META_KEY),
        file_field.metadata().get(PARQUET_FIELD_ID_META_KEY),
    ) {
        // A dropped and re-added column can reuse its name, but not its id.
        return table_id == file_id;
    }
    match case_sensitive {
        Some(true) => table_field.name() == file_field.name(),
        Some(false) => parquet_names_eq_ignore_case(table_field.name(), file_field.name()),
        None => table_field.name().eq_ignore_ascii_case(file_field.name()),
    }
}

/// Resolve predicates by the same identities as projected columns, before
/// Parquet statistics, Bloom filters or row filters see the physical schema.
#[derive(Debug)]
pub struct AuronPhysicalExprAdapterFactory(pub Option<bool>);

impl PhysicalExprAdapterFactory for AuronPhysicalExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Result<Arc<dyn PhysicalExprAdapter>> {
        Ok(Arc::new(AuronPhysicalExprAdapter {
            logical_file_schema,
            physical_file_schema,
            case_sensitive: self.0,
        }))
    }
}

#[derive(Debug)]
struct AuronPhysicalExprAdapter {
    logical_file_schema: SchemaRef,
    physical_file_schema: SchemaRef,
    case_sensitive: Option<bool>,
}

impl PhysicalExprAdapter for AuronPhysicalExprAdapter {
    fn rewrite(&self, expr: PhysicalExprRef) -> Result<PhysicalExprRef> {
        let adapter =
            AuronSchemaAdapter::new(self.logical_file_schema.clone(), self.case_sensitive);
        let rewrite_column = |expr: PhysicalExprRef| {
            let Some(column) = expr.downcast_ref::<Column>() else {
                return Ok(Transformed::no(expr));
            };
            let index = self.logical_file_schema.index_of(column.name())?;
            let field = self.logical_file_schema.field(index);
            let resolved = adapter
                .resolve_file_column(index, &self.physical_file_schema)?
                .map(|index| (index, self.physical_file_schema.field(index)));
            let mapped: PhysicalExprRef = match resolved {
                Some((index, file_field)) if file_field.data_type() == field.data_type() => {
                    Arc::new(Column::new(file_field.name(), index))
                }
                // Spark evaluates this conjunct after the scan mapper fills missing columns
                // and adapts types, including the layout of nested fields.
                _ if self.case_sensitive.is_some() => {
                    return Ok(Transformed::new(expr, false, TreeNodeRecursion::Stop));
                }
                Some((index, file_field)) => Arc::new(CastExpr::new(
                    Arc::new(Column::new(file_field.name(), index)),
                    field.data_type().clone(),
                    None,
                )),
                None => Arc::new(Literal::new(ScalarValue::try_from(field.data_type())?)),
            };
            Ok(Transformed::yes(mapped))
        };

        if self.case_sensitive.is_none() {
            return Ok(expr.transform_up(rewrite_column)?.data);
        }

        let mut predicates = Vec::new();
        for predicate in split_conjunction(&expr) {
            let rewritten = predicate.clone().transform_up(rewrite_column)?;
            if rewritten.tnr != TreeNodeRecursion::Stop {
                predicates.push(rewritten.data);
            }
        }
        Ok(conjunction(predicates))
    }
}

pub fn create_auron_schema_mapper(
    table_schema: &SchemaRef,
    field_mappings: &[Option<usize>],
) -> Arc<dyn SchemaMapper> {
    Arc::new(AuronSchemaMapping::new(
        table_schema.clone(),
        field_mappings.to_vec(),
        Arc::new(|col, field| schema_adapter_cast_column(col, field, None)),
    ))
}

fn schema_adapter_cast_column(
    col: &ArrayRef,
    field: &Field,
    case_sensitive: Option<bool>,
) -> Result<ArrayRef> {
    macro_rules! handle_decimal {
        ($s:ident, $t:ident, $tnative:ty, $prec:expr, $scale:expr) => {{
            use arrow::{array::*, datatypes::*};
            type DecimalBuilder = paste::paste! {[<$t Builder>]};
            type IntType = paste::paste! {[<$s Type>]};

            let col = col.as_primitive::<IntType>();
            let mut decimal_builder = DecimalBuilder::new();
            for i in 0..col.len() {
                if col.is_valid(i) {
                    decimal_builder.append_value(col.value(i) as $tnative);
                } else {
                    decimal_builder.append_null();
                }
            }
            Ok(Arc::new(
                decimal_builder
                    .finish()
                    .with_precision_and_scale($prec, $scale)?,
            ))
        }};
    }

    let col_dt = col.data_type();
    let cast_dt = field.data_type();
    let cast = || cast_scan_input_array(col.as_ref(), cast_dt, case_sensitive);

    match cast_dt {
        DataType::Decimal128(prec, scale) => match col_dt {
            DataType::Int8 => handle_decimal!(Int8, Decimal128, i128, *prec, *scale),
            DataType::Int16 => handle_decimal!(Int16, Decimal128, i128, *prec, *scale),
            DataType::Int32 => handle_decimal!(Int32, Decimal128, i128, *prec, *scale),
            DataType::Int64 => handle_decimal!(Int64, Decimal128, i128, *prec, *scale),
            DataType::Decimal128(..) => cast(),
            _ => df_execution_err!(
                "schema_adapter_cast_column unsupported type: {col_dt:?} => {cast_dt:?}",
            ),
        },
        DataType::List(to_field) => match col_dt {
            DataType::List(_from_field) => {
                let col = col.as_list::<i32>();
                let from_inner = col.values();
                let to_inner = schema_adapter_cast_column(from_inner, to_field, case_sensitive)?;
                Ok(Arc::new(ListArray::try_new(
                    to_field.clone(),
                    col.offsets().clone(),
                    to_inner,
                    col.nulls().cloned(),
                )?))
            }
            _ => df_execution_err!(
                "schema_adapter_cast_column unsupported type: {col_dt:?} => {cast_dt:?}",
            ),
        },
        _ => cast(),
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::Int32Array;

    use super::*;

    fn field_with_id(name: &str, id: &str) -> Field {
        Field::new(name, DataType::Int32, true)
            .with_metadata([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())].into())
    }

    #[test]
    fn test_field_ids_take_precedence_over_names() -> Result<()> {
        let table_schema = Arc::new(Schema::new(vec![field_with_id("value", "3")]));
        let adapter = AuronSchemaAdapter::new(table_schema.clone(), None);

        // A dropped and re-added column has the same name but a new identity.
        // If its field id is absent, the old values must not reappear.
        for has_matching_id in [false, true] {
            let mut fields = vec![field_with_id("value", "2")];
            let mut columns = vec![Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef];
            if has_matching_id {
                // A rename keeps its field id, even when another field has its name.
                fields.push(field_with_id("previous_name", "3"));
                columns.push(Arc::new(Int32Array::from(vec![30, 40])));
            }
            let file_schema = Arc::new(Schema::new(fields));
            let file_batch = RecordBatch::try_new(file_schema.clone(), columns)?;
            let (mapper, projection) = adapter.map_schema(&file_schema)?;
            assert_eq!(
                adapter.map_column_index(0, &file_schema),
                has_matching_id.then_some(1)
            );
            assert_eq!(projection, if has_matching_id { vec![1] } else { vec![] });
            let mapped = mapper.map_batch(file_batch.project(&projection)?)?;
            assert_eq!(mapped.schema(), table_schema);
            assert_eq!(mapped.num_rows(), 2);
            assert_eq!(
                mapped
                    .column(0)
                    .as_primitive::<arrow::datatypes::Int32Type>(),
                &Int32Array::from(if has_matching_id {
                    vec![Some(30), Some(40)]
                } else {
                    vec![None, None]
                })
            );
        }
        Ok(())
    }

    #[test]
    fn test_field_names_are_used_when_ids_are_unavailable() {
        for (table, file) in [
            (
                field_with_id("value", "3"),
                Field::new("VALUE", DataType::Int32, true),
            ),
            (
                Field::new("value", DataType::Int32, true),
                field_with_id("VALUE", "2"),
            ),
            (
                Field::new("value", DataType::Int32, true),
                Field::new("VALUE", DataType::Int32, true),
            ),
        ] {
            assert!(fields_match(&table, &file, None));
            assert!(!fields_match(&table, &file.with_name("different"), None));
        }
    }

    #[test]
    fn test_predicates_follow_field_ids_and_preserve_promoted_types() -> Result<()> {
        use datafusion::physical_expr::expressions::IsNullExpr;

        let logical_schema = Arc::new(Schema::new(vec![
            field_with_id("value", "3").with_data_type(DataType::Int64),
        ]));
        for has_matching_id in [false, true] {
            let mut fields = vec![field_with_id("value", "2")];
            let mut columns = vec![Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef];
            if has_matching_id {
                fields.push(field_with_id("previous_name", "3"));
                columns.push(Arc::new(Int32Array::from(vec![Some(30), None])));
            }
            let physical_schema = Arc::new(Schema::new(fields));
            let batch = RecordBatch::try_new(physical_schema.clone(), columns)?;
            let adapter = AuronPhysicalExprAdapterFactory(None)
                .create(logical_schema.clone(), physical_schema.clone())?;
            let column = Arc::new(Column::new("value", 0)) as PhysicalExprRef;
            let rewritten = adapter.rewrite(column.clone())?;
            assert_eq!(rewritten.data_type(&physical_schema)?, DataType::Int64);
            let values = rewritten.evaluate(&batch)?.into_array(2)?;
            assert_eq!(
                values.as_primitive::<arrow::datatypes::Int64Type>(),
                &arrow::array::Int64Array::from(if has_matching_id {
                    vec![Some(30), None]
                } else {
                    vec![None, None]
                }),
            );
            let predicate = adapter.rewrite(Arc::new(IsNullExpr::new(column)))?;
            let result = predicate.evaluate(&batch)?.into_array(2)?;
            assert_eq!(
                result.as_boolean(),
                &arrow::array::BooleanArray::from(vec![!has_matching_id, true]),
            );
        }
        Ok(())
    }

    #[test]
    fn test_parquet_predicate_rewrite_keeps_only_compatible_conjuncts() -> Result<()> {
        use datafusion::{
            logical_expr::Operator,
            physical_expr::expressions::{BinaryExpr, IsNullExpr},
        };

        let logical_schema = Arc::new(Schema::new(vec![
            Field::new("ID", DataType::Int32, true),
            Field::new("missing", DataType::Int32, true),
            Field::new("promoted", DataType::Int64, true),
        ]));
        let physical_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("ID", DataType::Int32, true),
            Field::new("promoted", DataType::Int32, true),
        ]));
        let adapter =
            AuronPhysicalExprAdapterFactory(Some(true)).create(logical_schema, physical_schema)?;
        let is_null = |name, index| {
            Arc::new(IsNullExpr::new(Arc::new(Column::new(name, index)))) as PhysicalExprRef
        };
        let unsupported = Arc::new(BinaryExpr::new(
            is_null("ID", 0),
            Operator::Or,
            is_null("missing", 1),
        )) as PhysicalExprRef;
        let predicate = conjunction([unsupported, is_null("promoted", 2), is_null("ID", 0)]);

        assert_eq!(
            adapter.rewrite(predicate)?.as_ref(),
            is_null("ID", 1).as_ref()
        );
        Ok(())
    }

    #[test]
    fn test_empty_projection_preserves_row_count() -> Result<()> {
        let schema = Arc::new(Schema::empty());
        let adapter = AuronSchemaAdapter::new(schema.clone(), None);
        let file_schema = Schema::new(vec![Field::new("value", DataType::Int32, true)]);
        let (mapper, projection) = adapter.map_schema(&file_schema)?;
        assert!(projection.is_empty());

        for num_rows in [0, 1, 1024] {
            let batch = RecordBatch::try_new_with_options(
                schema.clone(),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(num_rows)),
            )?;
            let mapped = mapper.map_batch(batch)?;
            assert_eq!(mapped.num_columns(), 0);
            assert_eq!(mapped.num_rows(), num_rows);
            assert_eq!(mapped.schema(), schema);
        }
        Ok(())
    }
}
