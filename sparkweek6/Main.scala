package com.example.spark.assignment

import java.sql.Timestamp

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.log4j.Logger
import org.apache.log4j.Level


object Main extends App {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  // Use this to your advantage
  case class StockPrice(ticker: String, close: Double, high: Double, low: Double, volume: Long, date: Timestamp)

  // Setup
  val sparkSession = SparkSession.builder().appName("assignment").master("local").getOrCreate()

  import sparkSession.implicits._

  // You can choose to use any one of these 3 distributed data structures
  val dataframe = sparkSession.read.option("header", "true").option("inferSchema", "true").csv("stock_prices.csv")
  val dataset = dataframe.as[StockPrice]
  val rdd = dataset.rdd

  dataframe.createOrReplaceTempView("stock_prices")
  dataframe.cache()
  val byDepName = Window.partitionBy('ticker).orderBy('date)
  val lagdataset = dataset.withColumn("lag", lag('close,1) over byDepName)
  val returndataset = lagdataset.withColumn("returns", ('close -'lag)/'lag)
  val volumeS = lagdataset.withColumn("volume$", ('close *'volume))
  
  val MeanReturnds = returndataset.groupBy("date").agg(avg('returns)).orderBy('date desc).show()
  val MeanVolumeS = volumeS.groupBy("ticker").agg(avg('volume$)).orderBy("avg(volume$)").show()
  val StdReturnds = returndataset.groupBy("ticker").agg(stddev_samp('returns)).orderBy("stddev_samp(returns)").show()
  
  
  

}
