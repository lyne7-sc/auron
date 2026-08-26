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

#![feature(test)]

extern crate test;

use std::sync::Arc;

use arrow::{
    array::{ArrayRef, Int32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};
use auron_memmgr::MemManager;
use datafusion::{
    common::JoinSide,
    physical_expr::{PhysicalExprRef, expressions::Column},
    physical_plan::{
        ExecutionPlan, common,
        joins::utils::{JoinOn, build_join_schema},
        test::TestMemoryExec,
    },
    prelude::SessionContext,
};
use datafusion_ext_plans::{
    broadcast_join_build_hash_map_exec::BroadcastJoinBuildHashMapExec,
    broadcast_join_exec::BroadcastJoinExec,
    joins::join_utils::JoinType::{self, LeftAnti, LeftSemi},
};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use test::{Bencher, black_box};

const PROBE_ROWS: usize = 100_000;
const BUILD_ROWS: usize = 10_000;

fn create_batch(keys: Vec<i32>) -> RecordBatch {
    let payloads = (0..keys.len() as i32).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(keys)),
        Arc::new(Int32Array::from(payloads)),
    ];
    RecordBatch::try_new(schema, columns).expect("failed to create benchmark batch")
}

fn create_probe_keys(hit_rate: usize) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(42 + hit_rate as u64);
    let num_matched = PROBE_ROWS * hit_rate / 100;
    let mut keys = Vec::with_capacity(PROBE_ROWS);
    keys.extend((0..num_matched).map(|_| rng.random_range(0..BUILD_ROWS as i32)));
    keys.extend(
        (num_matched..PROBE_ROWS)
            .map(|_| rng.random_range(BUILD_ROWS as i32..(BUILD_ROWS + PROBE_ROWS) as i32)),
    );
    keys.shuffle(&mut rng);
    keys
}

fn create_join(join_type: JoinType, hit_rate: usize) -> Arc<dyn ExecutionPlan> {
    let left_batch = create_batch(create_probe_keys(hit_rate));
    let right_batch = create_batch((0..BUILD_ROWS as i32).collect());
    let left_schema = left_batch.schema();
    let right_schema = right_batch.schema();
    let left: Arc<dyn ExecutionPlan> = Arc::new(
        TestMemoryExec::try_new(&[vec![left_batch]], left_schema.clone(), None)
            .expect("failed to create probe input"),
    );
    let right: Arc<dyn ExecutionPlan> = Arc::new(
        TestMemoryExec::try_new(&[vec![right_batch]], right_schema.clone(), None)
            .expect("failed to create build input"),
    );
    let left_key: PhysicalExprRef =
        Arc::new(Column::new_with_schema("key", &left_schema).expect("missing probe key column"));
    let right_key: PhysicalExprRef =
        Arc::new(Column::new_with_schema("key", &right_schema).expect("missing build key column"));
    let on: JoinOn = vec![(left_key, right_key.clone())];
    let right = Arc::new(BroadcastJoinBuildHashMapExec::new(right, vec![right_key]));
    let datafusion_join_type = join_type
        .try_into()
        .expect("failed to convert benchmark join type");
    let output_schema =
        Arc::new(build_join_schema(&left_schema, &right_schema, &datafusion_join_type).0);
    Arc::new(
        BroadcastJoinExec::try_new(
            output_schema,
            left,
            right,
            on,
            join_type,
            JoinSide::Right,
            true,
            None,
            false,
            None,
        )
        .expect("failed to create broadcast join"),
    )
}

fn bench_join(b: &mut Bencher, join_type: JoinType, hit_rate: usize) {
    MemManager::init(1 << 30);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to create benchmark runtime");
    let task_ctx = SessionContext::new().task_ctx();
    let join = create_join(join_type, hit_rate);

    b.iter(|| {
        let batches = runtime
            .block_on(async {
                let stream = join
                    .execute(0, task_ctx.clone())
                    .expect("failed to execute broadcast join");
                common::collect(stream).await
            })
            .expect("failed to collect broadcast join output");
        black_box(batches)
    });
}

macro_rules! benchmark {
    ($name:ident, $join_type:expr, $hit_rate:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            bench_join(b, $join_type, $hit_rate);
        }
    };
}

benchmark!(left_semi_10_percent_matches, LeftSemi, 10);
benchmark!(left_semi_50_percent_matches, LeftSemi, 50);
benchmark!(left_semi_90_percent_matches, LeftSemi, 90);
benchmark!(left_anti_10_percent_matches, LeftAnti, 10);
benchmark!(left_anti_50_percent_matches, LeftAnti, 50);
benchmark!(left_anti_90_percent_matches, LeftAnti, 90);
