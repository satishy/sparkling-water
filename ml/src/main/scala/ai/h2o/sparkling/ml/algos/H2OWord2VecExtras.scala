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
package ai.h2o.sparkling.ml.algos

import ai.h2o.sparkling.H2OFrame
import hex.word2vec.Word2VecModel.Word2VecParameters
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, explode, udf}

private[algos] trait H2OWord2VecExtras extends H2OAlgorithm[Word2VecParameters] {

  override protected def prepareH2OTrainFrameForFitting(trainFrame: H2OFrame): Unit = {
    super.prepareH2OTrainFrameForFitting(trainFrame)
    trainFrame.convertColumnsToStrings(Array(0))
  }

  private def addValue: UserDefinedFunction = udf((array: Seq[String]) => array ++ Array(""))

  override private[sparkling] def prepareDatasetForFitting(dataset: Dataset[_]): (H2OFrame, Option[H2OFrame]) = {
    if (getFeaturesCols().length != 1) {
      throw new IllegalArgumentException("Only one feature column is allowed as the input to H2OWord2Vec")
    }
    val featuresCol = getFeaturesCols().head
    val ds = dataset
      .withColumn(featuresCol, addValue(col(featuresCol)))
      .withColumn(featuresCol, explode(col(featuresCol)))
      .select(featuresCol)
    super.prepareDatasetForFitting(ds)
  }
}
