package sparkweek7
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.mllib.rdd.MLPairRDDFunctions.fromPairRDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.mllib._
import org.apache.spark.sql.DataFrameNaFunctions
import org.apache.spark.sql.functions._
import org.apache.spark.sql
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
object HealthCare {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark: SparkSession = SparkSession.builder.master("local").getOrCreate
    val sc = spark.sparkContext
    
    val train =  spark.read.option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", true)
      .csv("../train_2v.csv")
      
    val test =  spark.read.option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", true)
      .csv("../test_2v.csv") 
      
    train.show(10)
    train.printSchema()
    train.createOrReplaceTempView("table")
    val showWork = spark.sql("SELECT work_type, count(work_type) as work_type_count FROM table WHERE stroke == 1 GROUP BY work_type ORDER BY work_type_count DESC").show()
    val showParticip = spark.sql("SELECT gender, count(gender) as count_gender, count(gender)*100/sum(count(gender)) over() as percent FROM table GROUP BY gender").show()
    val train_f = train.na.fill("No Info", Seq("smoking_status"))
    val tmean = train_f.select(avg(train_f("bmi"))).collect()
    val mean_bmi = tmean(0)(0).asInstanceOf[Double]
    val train_f2 = train_f.na.fill(mean_bmi, Seq("bmi"))
    
    val gender_indexer = new StringIndexer().setInputCol("gender").setOutputCol("genderIndex")
    val gender_encoder = new OneHotEncoder().setInputCol("genderIndex").setOutputCol("genderVec")
    
    val ever_married_indexer = new StringIndexer().setInputCol("ever_married").setOutputCol("ever_marriedIndex")
    val ever_married_encoder = new OneHotEncoder().setInputCol("ever_marriedIndex").setOutputCol("ever_marriedVec")
    
    val work_type_indexer = new StringIndexer().setInputCol("work_type").setOutputCol("work_typeIndex")
    val work_type_encoder = new OneHotEncoder().setInputCol("work_typeIndex").setOutputCol("work_typeVec")
    
    val smoking_status_indexer = new StringIndexer().setInputCol("smoking_status").setOutputCol("smoking_statusIndex")
    val smoking_status_encoder = new OneHotEncoder().setInputCol("smoking_statusIndex").setOutputCol("smoking_statusVec")
    
    val Residence_type_indexer = new StringIndexer().setInputCol("Residence_type").setOutputCol("Residence_typeIndex")
    val Residence_type_encoder = new OneHotEncoder().setInputCol("Residence_typeIndex").setOutputCol("Residence_typeVec")

    val assembler = new VectorAssembler()
    .setInputCols(Array("genderVec",
                         "age",
                         "hypertension",
                         "heart_disease",
                         "ever_marriedVec",
                         "work_typeVec",
                         "Residence_typeVec",
                         "avg_glucose_level",
                         "bmi",
                         "smoking_statusVec"))
                         .setOutputCol("features")
                         
  val dtc = new DecisionTreeClassifier().setLabelCol("stroke").setFeaturesCol("features")

  
  val pipeline = new Pipeline().setStages(Array(gender_indexer, ever_married_indexer, work_type_indexer, Residence_type_indexer,
                           smoking_status_indexer, gender_encoder, ever_married_encoder, work_type_encoder,
                           Residence_type_encoder, smoking_status_encoder, assembler, dtc))
                        
  val split = train_f2.randomSplit(Array(0.7, 0.3), seed = 11L)
  val train_data = split(0).cache()
  val test_data = split(1).cache()
  val model = pipeline.fit(train_data)
  val predictions = model.transform(test_data)
  
  predictions.show(5)
  
  val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("stroke")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
  
  val accuracy = evaluator.evaluate(predictions)
  println(s"Test Error = ${(1.0 - accuracy)}")
  
  
    
  }}   
