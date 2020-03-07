package sparkweek7

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
/** Count up how many of each word appears in a book as simply as possible. */
object WordCount {
 

/** Our main function where the action happens */
  def main(args: Array[String]) {
   
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark: SparkSession = SparkSession.builder.master("local").getOrCreate

    //val conf = new  SparkConf().setMaster("local[*]").setAppName("WordCount_v3").set("spark.driver.host", "localhost");
    // Create a SparkContext using every core of the local machine, named WordCountBetterSorted
    //alternative: val sc = new SparkContext("local[*]", "WordCount_v3")
    val sc = spark.sparkContext
    
    // Load each line of my book into an RDD
    val input = sc.textFile("../book.txt")
    
    // Split using a regular expression that extracts words
    val words = input.flatMap(x => x.split("\\W+"))
    
    // Normalize everything to lowercase
    val lowercaseWords = words.map(x => x.toLowerCase())
    
    // Count of the occurrences of each word
    val wordCounts = lowercaseWords.map(x => (x, 1)).reduceByKey( (x,y) => x + y )
    
    // Flip (word, count) tuples to (count, word) and then sort by key (the counts)
    val wordCountsSorted = wordCounts.map( x => (x._2, x._1) ).sortByKey().collect
    
    // Print the results, flipping the (count, word) results to word: count as we go.
    //for (result <- wordCountsSorted) {
      //val count = result._1
      //val word = result._2
      //println(s"$word: $count")}
      
    val df = spark.createDataFrame(wordCountsSorted).toDF( "count","word")
    
    val genword = List("you", "to", "your", "the", "a", "of", "and")
    val df2 = df.filter(!df("word").isin(genword : _*)) 
    val df3 = df2.orderBy(df2("count").desc).show(10)
    
    
  } 
}

