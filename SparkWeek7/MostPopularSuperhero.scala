package sparkweek7

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.mllib.rdd.MLPairRDDFunctions.fromPairRDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
/** Find the superhero with the most co-appearances. */
object MostPopularSuperhero {
  
  // Function to extract the hero ID and number of connections from each line
  def countCoOccurences(line: String) = {
    var elements = line.split("\\s+")
    ( elements(0).toInt, elements.length - 1 )
  }
  
  // Function to extract hero ID -> hero name tuples (or None in case of failure)
  def parseNames(line: String) : Option[(Int, String)] = {
    var fields = line.split('\"')
    if (fields.length > 1) {
      return Some(fields(0).trim().toInt, fields(1))
    } else {
      return None // flatmap will just discard None results, and extract data from Some results.
    }
  }
 
  /** Our main function where the action happens */
  def main(args: Array[String]) {
   
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark: SparkSession = SparkSession.builder.master("local").getOrCreate

    val conf = new  SparkConf().setMaster("local[*]").setAppName("MostPopularSuperhero").set("spark.driver.host", "localhost");
    // Create a SparkContext using every core of the local machine, named MostPopularSuperhero
    //alternative: val sc = new SparkContext("local[*]", "MostPopularSuperhero")
    //val sc = new SparkContext(conf)   
     val sc = spark.sparkContext
    // Build up a hero ID -> name RDD
    val names = sc.textFile("../marvel-names.txt")
    val namesRdd = names.flatMap(parseNames)
    
    // Load up the superhero co-apperarance data
    val lines = sc.textFile("../marvel-graph.txt")
    
    // Convert to (heroID, number of connections) RDD
    val pairings = lines.map(countCoOccurences)
    
    // Combine entries that span more than one line
    val totalFriendsByCharacter = pairings.reduceByKey( (x,y) => x + y )
    
    // Flip it to # of connections, hero ID
    val flipped = totalFriendsByCharacter.map( x => (x._2, x._1) )
    
    // Find the max # of connections
    val mostPopular = flipped.max()
    
    // Look up the name (lookup returns an array of results, so we need to access the first result with (0)).
    val mostPopularName = namesRdd.lookup(mostPopular._2)(0)
    
    // Print out our answer!
    println(s"$mostPopularName is the most popular superhero with ${mostPopular._1} co-appearances.") 
    
   // val result = flipped.sortBy(_._1,false).top(10)
    val jointresult = flipped.sortBy(_._1,false).top(10)
    jointresult.foreach(x => println((x._1, x._2)))
    val df = spark.createDataFrame(jointresult).toDF( "count","heronum")
    val dfname = spark.createDataFrame(namesRdd).toDF( "heronum","name")
    val joindf = df.join(dfname, df.col("heronum").equalTo(dfname("heronum")))
      .drop(df.col("heronum"))
      .drop(dfname.col("heronum"))
      .orderBy(df("count").desc).show();
     
     
  }
  
}
