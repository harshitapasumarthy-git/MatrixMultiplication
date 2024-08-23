// Databricks notebook source
// MAGIC %md
// MAGIC **Main Project (100 pts)** \
// MAGIC Implement closed-form solution when m(number of examples is large) and n(number of features) is small:
// MAGIC \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\]
// MAGIC Here, X is a distributed matrix.

// COMMAND ----------

// MAGIC %md
// MAGIC Steps:
// MAGIC 1. Create an example RDD for matrix X of type RDD\[Int, Int, Double\] and vector y of type RDD\[Int, Double\]
// MAGIC 2. Compute \\[ \scriptsize \mathbf{(X^TX)}\\]
// MAGIC 3. Convert the result matrix to a Breeze Dense Matrix and compute pseudo-inverse
// MAGIC 4. Compute \\[ \scriptsize \mathbf{X^Ty}\\] and convert it to Breeze Vector
// MAGIC 5. Multiply \\[ \scriptsize \mathbf{(X^TX)}^{-1}\\] with \\[ \scriptsize \mathbf{X^Ty}\\]

// COMMAND ----------

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import breeze.linalg.{DenseMatrix, pinv,inv,DenseVector}



// COMMAND ----------

 // Create an example RDD for matrix X of type RDD[Int, Int, Double] and vector y of type RDD[Int, Double]
val x = Seq(
  (0, 0, 1.0),
  (1, 1, 4.0),
  (2, 2, 7.0),
)
val X: RDD[(Int, Int, Double)] = sc.parallelize(x)
val y = Seq(
  (0, 1.0),
  (1, 2.0),
  (2, 3.0)
)

val Y: RDD[(Int, Double)] = sc.parallelize(y)




// COMMAND ----------


//  Compute(XTX)
val XTranspose = X.map { case (i, j, value) => (j, i, value) }
val XTransposeX = XTranspose
  .map { case (j, i, value) => ((i, j), value) }
  .join(X.map { case (i, j, value) => ((i, j), value) })
  .map { case ((i, j), (value1, value2)) => ((i, j), value1 * value2) }
  .reduceByKey(_ + _)







// COMMAND ----------

// Convert to breeze dense matrix
//  Collect data from RDD
val resultXTX = XTransposeX.collect()
val numRows = resultXTX.map(_._1._1).max + 1
val numCols = resultXTX.map(_._1._2).max + 1
val matrixData = Array.ofDim[Double](numRows, numCols)
resultXTX.foreach { case ((i, j), value) =>
  matrixData(i)(j) = value
}
val denseMatrixXTX = DenseMatrix(matrixData: _*)

// Compute pseudo-inverse using Breeze
val pseudoInverse = pinv(denseMatrixXTX)
println("Pseudo-Inverse:\n" + pseudoInverse)

// COMMAND ----------


val XT_y = XTranspose
  .map { case (j, i, value) => (i, (j, value)) }
  .join(Y)
  .map { case (i, ((j, xValue), yValue)) => (j, xValue * yValue) }
  .reduceByKey(_ + _)
  val collectedXT_y = XT_y.collect()
val xtYMap = collectedXT_y.toMap
val breezeVectorXT_y = new DenseVector((0 to xtYMap.keys.max).map(i => xtYMap.getOrElse(i, 0.0)).toArray)


// COMMAND ----------

println(s"pseudo-inverse matrix: ${pseudoInverse.rows}x${pseudoInverse.cols}")
println(s"XT_y vector: ${breezeVectorXT_y.length}")

// Multiply (X^T X)^-1 with X^T y
val result = pseudoInverse * breezeVectorXT_y
println("Result:\n" + result)



// COMMAND ----------

// MAGIC %md
// MAGIC **Bonus 1(10 pts)** \
// MAGIC Implement \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\] using Spark DataFrame.  
// MAGIC
// MAGIC Note: Your queries should be in the following format:
// MAGIC \\[ \scriptsize \mathbf{spark.sql("select ... from ...")}\\]

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SparkSession, DataFrame}
import spark.implicits._

// Matrix `X` as DataFrame
val x = Seq(
  (0, 0, 1.0),
  (1, 1, 4.0),
  (2, 2, 7.0)
).toDF("row", "col", "value")

// Vector `y` as DataFrame
val y = Seq(
  (0, 1.0),
  (1, 2.0),
  (2, 3.0)
).toDF("index", "value")

x.createOrReplaceTempView("X")
y.createOrReplaceTempView("y")


// COMMAND ----------


// Compute X^T (transpose of X)
val xt = spark.sql("""
  SELECT col AS row, row AS col, value
  FROM X
""")
xt.createOrReplaceTempView("XT")


// Compute X^T X
val xt_x = spark.sql("""
  SELECT a.col AS row, b.col AS col, SUM(a.value * b.value) AS value
  FROM XT a
  JOIN X b ON a.row = b.row
  GROUP BY a.col, b.col
""")
xt_x.createOrReplaceTempView("XT_X")

// Collect the values and convert to Breeze matrix
val numFeatures = xt_x.select("row").distinct().count().toInt

val xtxArray = xt_x.collect().map {
  case Row(row: Int, col: Int, value: Double) => ((row, col), value)
}.toMap

val xtxMatrix = DenseMatrix.tabulate[Double](numFeatures, numFeatures) { (i, j) =>
  xtxArray.getOrElse((i, j), 0.0)
}
// Invert the matrix using Breeze
val xtxInvMatrix = inv(xtxMatrix)
val xtxInvSeq = for {
  i <- 0 until xtxInvMatrix.rows
  j <- 0 until xtxInvMatrix.cols
} yield (i, j, xtxInvMatrix(i, j))

val xtxInvDF = xtxInvSeq.toDF("row", "col", "value")

xtxInvDF.createOrReplaceTempView("XtXInv")

xtxInvDF.show()

// COMMAND ----------

// Compute X^T y
val xt_y = spark.sql("""
  SELECT a.col AS row, SUM(a.value * b.value) AS value
  FROM X a
  JOIN y b ON a.row = b.index
  GROUP BY a.col
""")

xt_y.createOrReplaceTempView("XT_y")


// COMMAND ----------

// Compute theta 
val theta = spark.sql("""
  SELECT a.row, SUM(a.value * b.value) AS theta
  FROM XtXInv a
  JOIN XT_y b ON a.row = b.row
  GROUP BY a.row
""")

theta.show()


// COMMAND ----------

// MAGIC %md
// MAGIC

// COMMAND ----------

// MAGIC %md
// MAGIC **Bonus 2(10 pts)** \
// MAGIC Run both of your implementations (main project using RDDs, bonus 1 using Dataframes) on Boston Housing Dataset: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices?resource=download. Which implementation performs better?
