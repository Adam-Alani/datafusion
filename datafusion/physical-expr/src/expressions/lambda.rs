// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Physical lambda expression: [`LambdaExpr`]

use std::hash::Hash;
use std::sync::Arc;

use crate::{
    ScalarFunctionExpr,
    expressions::{Column, LambdaVariable},
    physical_expr::PhysicalExpr,
};
use arrow::{
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
use datafusion_common::{
    HashMap, plan_err,
    tree_node::{Transformed, TreeNode, TreeNodeRecursion},
};
use datafusion_common::{HashSet, Result, internal_err};
use datafusion_expr::ColumnarValue;

/// Represents a lambda with the given parameters names and body
#[derive(Debug, Eq, Clone)]
pub struct LambdaExpr {
    params: Vec<String>,
    body: Arc<dyn PhysicalExpr>,
    projected_body: Arc<dyn PhysicalExpr>,
    projection: Vec<usize>,
    /// Number of columns in the outer input schema. Column/LambdaVariable
    /// indices below this value reference outer captures; indices at or above
    /// reference lambda parameters (whose position in the merged evaluation
    /// batch is fixed by the higher-order function, not by the projection).
    outer_columns_count: usize,
}

// Manually derive PartialEq and Hash to work around https://github.com/rust-lang/rust/issues/78808 [https://github.com/apache/datafusion/issues/13196]
impl PartialEq for LambdaExpr {
    fn eq(&self, other: &Self) -> bool {
        self.params.eq(&other.params) && self.body.eq(&other.body)
    }
}

impl Hash for LambdaExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.params.hash(state);
        self.body.hash(state);
    }
}

impl LambdaExpr {
    /// Create a new lambda expression with the given parameters and body.
    ///
    /// `outer_columns_count` is the number of columns in the outer input
    /// schema this lambda was planned against. Column/LambdaVariable indices
    /// strictly below `outer_columns_count` reference outer captures and get
    /// compressed to the front of the evaluation batch; indices at or above
    /// reference lambda parameters and keep their fixed position relative to
    /// the captures (so unused lambda parameters do not shift used ones).
    pub fn try_new(
        params: Vec<String>,
        body: Arc<dyn PhysicalExpr>,
        outer_columns_count: usize,
    ) -> Result<Self> {
        if !all_unique(&params) {
            return plan_err!(
                "lambda params must be unique, got ({})",
                params.join(", ")
            );
        }

        check_async_udf(&body)?;

        Ok(Self::new(params, body, outer_columns_count))
    }

    fn new(
        params: Vec<String>,
        body: Arc<dyn PhysicalExpr>,
        outer_columns_count: usize,
    ) -> Self {
        let mut used_column_indices = HashSet::new();

        body.apply(|node| {
            if let Some(col) = node.downcast_ref::<Column>() {
                used_column_indices.insert(col.index());
            } else if let Some(var) = node.downcast_ref::<LambdaVariable>() {
                used_column_indices.insert(var.index());
            }

            Ok(TreeNodeRecursion::Continue)
        })
        .expect("closure should be infallible");

        let mut projection = used_column_indices.into_iter().collect::<Vec<_>>();

        projection.sort();

        // Captures (outer column refs) get compressed to the front of the
        // merged batch. Lambda variables (indices >= outer_columns_count)
        // keep their fixed offset from the start of the lambda parameter
        // block, because the higher-order function always pushes all
        // declared parameters into the merged batch in order.
        let used_captures_count = projection
            .iter()
            .filter(|i| **i < outer_columns_count)
            .count();
        let column_index_map = projection
            .iter()
            .enumerate()
            .map(|(captures_pos, original)| {
                let projected = if *original < outer_columns_count {
                    captures_pos
                } else {
                    used_captures_count + (*original - outer_columns_count)
                };
                (*original, projected)
            })
            .collect::<HashMap<_, _>>();

        let projected_body = Arc::clone(&body)
            .transform_down(|e| {
                if let Some(column) = e.downcast_ref::<Column>() {
                    let original = column.index();
                    let projected = *column_index_map.get(&original).unwrap();
                    if projected != original {
                        return Ok(Transformed::yes(Arc::new(Column::new(
                            column.name(),
                            projected,
                        ))));
                    }
                } else if let Some(lambda_variable) = e.downcast_ref::<LambdaVariable>() {
                    let original = lambda_variable.index();
                    let projected = *column_index_map.get(&original).unwrap();
                    if projected != original {
                        return Ok(Transformed::yes(Arc::new(LambdaVariable::new(
                            projected,
                            Arc::clone(lambda_variable.field()),
                        ))));
                    }
                }
                Ok(Transformed::no(e))
            })
            .expect("closure should be infallible")
            .data;

        Self {
            params,
            body,
            projected_body,
            projection,
            outer_columns_count,
        }
    }

    /// Get the lambda's params names
    pub fn params(&self) -> &[String] {
        &self.params
    }

    /// Get the lambda's body
    pub fn body(&self) -> &Arc<dyn PhysicalExpr> {
        &self.body
    }

    pub(crate) fn projection(&self) -> &[usize] {
        &self.projection
    }

    pub(crate) fn projected_body(&self) -> &Arc<dyn PhysicalExpr> {
        &self.projected_body
    }
}

impl std::fmt::Display for LambdaExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}) -> {}", self.params.join(", "), self.body)
    }
}

impl PhysicalExpr for LambdaExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Null)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, _batch: &RecordBatch) -> Result<ColumnarValue> {
        internal_err!("LambdaExpr::evaluate() should not be called")
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.body]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let [body] = children.as_slice() else {
            return internal_err!(
                "LambdaExpr expects exactly 1 child, got {}",
                children.len()
            );
        };

        check_async_udf(body)?;

        Ok(Arc::new(Self::new(
            self.params.clone(),
            Arc::clone(body),
            self.outer_columns_count,
        )))
    }

    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) -> {}", self.params.join(", "), self.body)
    }
}

/// Create a lambda expression.
///
/// `outer_columns_count` is the number of columns in the outer input schema
/// this lambda was planned against. See [`LambdaExpr::try_new`] for details.
pub fn lambda(
    params: impl IntoIterator<Item = impl Into<String>>,
    body: Arc<dyn PhysicalExpr>,
    outer_columns_count: usize,
) -> Result<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(LambdaExpr::try_new(
        params.into_iter().map(Into::into).collect(),
        body,
        outer_columns_count,
    )?))
}

fn all_unique(params: &[String]) -> bool {
    match params.len() {
        0 | 1 => true,
        2 => params[0] != params[1],
        _ => {
            let mut set = HashSet::with_capacity(params.len());

            params.iter().all(|p| set.insert(p.as_str()))
        }
    }
}

fn check_async_udf(body: &Arc<dyn PhysicalExpr>) -> Result<()> {
    if body.exists(|expr| {
        Ok(expr
            .downcast_ref::<ScalarFunctionExpr>()
            .is_some_and(|udf| udf.fun().as_async().is_some()))
    })? {
        return plan_err!(
            "Async functions in lambdas aren't supported, see https://github.com/apache/datafusion/issues/22091"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::expressions::{Column, LambdaVariable, NoOp, lambda::lambda};
    use arrow::{
        array::RecordBatch,
        datatypes::{DataType, Field, Schema},
    };
    use std::sync::Arc;

    use super::LambdaExpr;

    #[test]
    fn test_lambda_evaluate() {
        let lambda = lambda(["a"], Arc::new(NoOp::new()), 0).unwrap();
        let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
        assert!(lambda.evaluate(&batch).is_err());
    }

    #[test]
    fn test_lambda_duplicate_name() {
        assert!(lambda(["a", "a"], Arc::new(NoOp::new()), 0).is_err());
    }

    /// Multi-parameter lambdas that reference only a subset of their declared
    /// parameters must NOT shift the used parameter into the slot of an
    /// unused one. The higher-order function pushes all declared parameters
    /// into the merged batch in order, so a lambda `(k, v) -> v` (with `v` at
    /// LambdaVariable index 1) must keep its body referencing index 1, not
    /// get re-projected to index 0 just because `k` is unused.
    #[test]
    fn test_multi_param_lambda_keeps_param_positions_stable() {
        let v_field = Arc::new(Field::new("v", DataType::Int32, true));
        let body = Arc::new(LambdaVariable::new(1, Arc::clone(&v_field)));

        let lambda = LambdaExpr::try_new(
            vec!["k".to_string(), "v".to_string()],
            body,
            0, // no outer captures
        )
        .unwrap();

        assert_eq!(lambda.projection(), &[1]);

        let projected_var = lambda
            .projected_body()
            .downcast_ref::<LambdaVariable>()
            .expect("projected body should be a LambdaVariable");
        assert_eq!(projected_var.index(), 1);
    }

    /// With outer captures, used outer columns get compressed to the front of
    /// the projected batch, but lambda parameter positions stay at their
    /// fixed offset from the start of the lambda parameter block (so an
    /// unused parameter still leaves a gap rather than shifting later ones).
    #[test]
    fn test_lambda_compresses_outer_captures_but_pins_params() {
        // outer schema has 5 columns (indices 0..=4); lambda has 2 params at
        // indices 5 and 6. Body references outer column 3 and the second
        // lambda param (`v` at index 6); the first lambda param (`k` at 5)
        // is unused.
        let v_field = Arc::new(Field::new("v", DataType::Int32, true));
        let body = Arc::new(crate::expressions::BinaryExpr::new(
            Arc::new(Column::new("c3", 3)),
            datafusion_expr::Operator::Plus,
            Arc::new(LambdaVariable::new(6, Arc::clone(&v_field))),
        ));

        let lambda = LambdaExpr::try_new(
            vec!["k".to_string(), "v".to_string()],
            body,
            5, // outer_columns_count
        )
        .unwrap();

        // Both originals are referenced (3 and 6), projection is sorted.
        assert_eq!(lambda.projection(), &[3, 6]);

        // After projection:
        //   outer col 3   -> position 0 (compressed to the captures block)
        //   lambda var 6  -> position 2 (used_captures=1 + (6 - 5))
        //                    NOT position 1, because the unused `k` (var 5)
        //                    still occupies its slot.
        let binary = lambda
            .projected_body()
            .downcast_ref::<crate::expressions::BinaryExpr>()
            .expect("projected body should be a BinaryExpr");
        let left = binary
            .left()
            .downcast_ref::<Column>()
            .expect("left should be a Column");
        let right = binary
            .right()
            .downcast_ref::<LambdaVariable>()
            .expect("right should be a LambdaVariable");
        assert_eq!(left.index(), 0);
        assert_eq!(right.index(), 2);
    }
}
