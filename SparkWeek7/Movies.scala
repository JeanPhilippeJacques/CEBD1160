package sparkweek7
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._ 
object Movies {
   def main(args: Array[String]) {
   
   
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark: SparkSession = SparkSession.builder.master("local").getOrCreate
  
    val sc = spark.sparkContext
    // Read in each rating line
    val lines = sc.textFile("src/sparkweek7/u.data")
    val linesdf = spark.read.option("delimiter", "|").csv("src/sparkweek7/u.item")
    val linesdf2=linesdf.withColumnRenamed("_c0","items")
    
    
    val myFile = sc.textFile("src/sparkweek7/u.data")
  
    val df= myFile.map(_.split("\t")).map{case Array(a,b,c,d) => (a,b,c,d)}
    val rdf = spark.createDataFrame(df).toDF( "users","items","ratings","unk")
    
    val rdf2 = rdf.groupBy("items")
            .agg(count("items").alias("count"),
                mean("ratings").alias("Mean")
                )
            .filter("count >= 200")
            
     val rdf3 =rdf2.orderBy(rdf2("Mean").desc)
     val output = rdf3.join(linesdf2,Seq("items"),joinType="inner").show(10)   
    
  }
}