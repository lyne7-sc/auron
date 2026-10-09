/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.auron

import org.apache.spark.sql.AuronQueryTest
import org.apache.spark.sql.execution.auron.plan.NativeWindowBase

import org.apache.auron.util.AuronTestUtils

class AuronWindowSuite extends AuronQueryTest with BaseAuronSQLSuite with AuronSQLTestHelper {

  test("aggregate window functions with cumulative ROWS frame") {
    withSQLConf(
      "spark.auron.enable.window" -> "true",
      "spark.sql.ansi.enabled" -> "false",
      "spark.sql.adaptive.enabled" -> "false",
      "spark.auron.batchSize" -> "2") {
      withTable("t1") {
        sql("create table t1(id int, grp int, v int) using parquet")
        sql("""insert into t1 values
            |(1, 1, null), (2, 1, 4), (3, 1, 2), (4, 1, null),
            |(5, 2, 8), (6, 2, null), (7, 3, null), (8, 3, null)
            |""".stripMargin)

        for {
          partition <- Seq("partition by grp", "")
          value <- Seq("v", "cast(v as double)")
        } {
          val frame = s"$partition order by id rows between unbounded preceding and current row"
          val df = checkSparkAnswer(s"""select id, grp,
              |sum($value) over ($frame) as sum_v,
              |avg($value) over ($frame) as avg_v,
              |min($value) over ($frame) as min_v,
              |max($value) over ($frame) as max_v,
              |count($value) over ($frame) as count_v,
              |count(*) over ($frame) as count_all
              |from t1
              |""".stripMargin)
          val plan = stripAQEPlan(df.queryExecution.executedPlan)
          assert(plan.collectFirst { case _: NativeWindowBase => true }.isDefined, plan.toString)
        }
      }
    }
  }

  test("decimal sum and average window functions fall back") {
    withSQLConf("spark.auron.enable.window" -> "true", "spark.sql.ansi.enabled" -> "false") {
      withTable("t1") {
        sql("create table t1(id int, v decimal(38, 0)) using parquet")
        sql("insert into t1 values (1, 1), (2, 2), (3, 2)")
        for (function <- Seq("sum", "avg")) {
          val df = checkSparkAnswer(s"""select id,
              |$function(v) over (
              |order by id rows between unbounded preceding and current row) as result
              |from t1
              |""".stripMargin)
          val plan = stripAQEPlan(df.queryExecution.executedPlan)
          assert(plan.collectFirst { case _: NativeWindowBase => true }.isEmpty, plan.toString)
        }
      }
    }
  }

  test("aggregate window functions with unsupported frames fall back") {
    withSQLConf("spark.auron.enable.window" -> "true", "spark.sql.ansi.enabled" -> "false") {
      withTable("t1") {
        sql("create table t1(id int, v int) using parquet")
        sql("insert into t1 values (1, 2), (1, 3), (2, null), (3, 4)")
        for (frame <- Seq(
            "range between unbounded preceding and current row",
            "rows between 1 preceding and current row")) {
          val df = checkSparkAnswer(s"""select id,
              |sum(v) over (order by id $frame) as sum_v
              |from t1
              |""".stripMargin)
          val plan = stripAQEPlan(df.queryExecution.executedPlan)
          assert(plan.collectFirst { case _: NativeWindowBase => true }.isEmpty, plan.toString)
        }
      }
    }
  }

  test("ANSI sum and average window functions fall back") {
    withSQLConf("spark.auron.enable.window" -> "true", "spark.sql.ansi.enabled" -> "true") {
      withTable("t1") {
        sql("create table t1(id int, v int) using parquet")
        sql("insert into t1 values (1, 2), (2, null), (3, 4)")
        for (function <- Seq("sum", "avg")) {
          val df = checkSparkAnswer(s"""select id,
              |$function(v) over (
              |order by id rows between unbounded preceding and current row) as result
              |from t1
              |""".stripMargin)
          val plan = stripAQEPlan(df.queryExecution.executedPlan)
          assert(plan.collectFirst { case _: NativeWindowBase => true }.isEmpty, plan.toString)
        }
      }
    }
  }

  test("TRY sum and average window functions fall back") {
    if (AuronTestUtils.isSparkV33OrGreater) {
      withSQLConf("spark.auron.enable.window" -> "true", "spark.sql.ansi.enabled" -> "false") {
        withTable("t1") {
          sql("create table t1(id int, v bigint) using parquet")
          sql("insert into t1 values (1, 9223372036854775807), (2, 1)")
          for (function <- Seq("try_sum", "try_avg")) {
            val df = checkSparkAnswer(s"""select id,
                |$function(v) over (
                |order by id rows between unbounded preceding and current row) as result
                |from t1
                |""".stripMargin)
            val plan = stripAQEPlan(df.queryExecution.executedPlan)
            assert(plan.collectFirst { case _: NativeWindowBase => true }.isEmpty, plan.toString)
          }
        }
      }
    }
  }

  test("lead window function") {
    withSQLConf("spark.auron.enable.window" -> "true") {
      withTable("t1") {
        sql("create table t1(id int, grp int, v string) using parquet")
        sql("insert into t1 values (1, 1, 'a'), (2, 1, null), (3, 1, 'c'), (4, 2, 'x')")

        checkSparkAnswerAndOperator("""select
            |  id,
            |  grp,
            |  v,
            |  lead(v) over (partition by grp order by id) as next_v,
            |  lead(v, 2, 'fallback') over (partition by grp order by id) as next2_v
            |from t1
            |""".stripMargin)
      }
    }
  }

  test("lead window function with ignore nulls falls back") {
    if (AuronTestUtils.isSparkV32OrGreater) {
      withSQLConf("spark.auron.enable.window" -> "true") {
        withTable("t1") {
          sql("create table t1(id int, grp int, v string) using parquet")
          sql("insert into t1 values (1, 1, 'a'), (2, 1, null), (3, 1, 'c'), (4, 2, 'x')")

          val df = checkSparkAnswer("""select
              |  id,
              |  grp,
              |  lead(v, 1, 'fallback') ignore nulls
              |    over (partition by grp order by id) as next_non_null_v
              |from t1
              |""".stripMargin)
          val plan = stripAQEPlan(df.queryExecution.executedPlan)
          assert(plan.collectFirst { case _: NativeWindowBase => true }.isEmpty)
        }
      }
    }
  }
}
