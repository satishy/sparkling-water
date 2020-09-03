package ai.h2o.sparkling.ml.features

import ai.h2o.sparkling.H2OFrame
import ai.h2o.sparkling.ml.algos.H2OAlgorithm
import hex.word2vec.Word2VecModel.Word2VecParameters
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, explode, udf}

private[features] trait H2OWord2VecExtras extends H2OAlgorithm[Word2VecParameters] {

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
