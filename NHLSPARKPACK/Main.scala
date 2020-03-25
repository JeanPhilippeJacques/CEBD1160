package NHLSPARKPACK
import java.sql.Timestamp
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.mllib._
import org.apache.spark.mllib.rdd.MLPairRDDFunctions.fromPairRDD
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
object Main  extends App {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("org").setLevel(Level.ERROR)
    val sparkSession = SparkSession.builder().appName("NHL").master("local[*]").config("spark.driver.host", "localhost").getOrCreate()
  //.config("spark.driver.bindAddress", "127.0.0.1")
    import sparkSession.implicits._
    
    //Import Data
    val player  =  sparkSession.read.option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", true)
      .csv("../NHLSPARK/src/NHLSPARKPACK/player_info.csv")
      
    val team=  sparkSession.read.option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", true)
      .csv("../NHLSPARK/src/NHLSPARKPACK/team_info.csv") 
    
     val skater_stats =  sparkSession.read.option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", true)
      .csv("../NHLSPARK/src/NHLSPARKPACK/game_skater_stats.csv") 
      
     val game =  sparkSession.read.option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", true)
      .csv("../NHLSPARK/src/NHLSPARKPACK/game.csv") 
      
      //Join Data
      val SS_Game = skater_stats.join(game, skater_stats.col("game_id").equalTo(game("game_id")))
      .drop(game.col("game_id"))
      .orderBy(game("date_time"))
     
      val SS_Game_Pyer = SS_Game.join(player, SS_Game.col("player_id").equalTo(player("player_id")))
      .drop(player.col("player_id"))
      .orderBy(SS_Game("date_time").desc).filter("type == 'R'")
      SS_Game_Pyer.cache()
      
      //Lag stats over the last 10 20 games 
      val last10 = Window.partitionBy('player_id).orderBy('date_time).rowsBetween(-11, -1)
      val last20 = Window.partitionBy('player_id).orderBy('date_time).rowsBetween(-21, -1)
      val lagdataset = SS_Game_Pyer.withColumn("l10_assists", sum('assists) over last10).withColumn(
                                               "L10_goals", sum('goals) over last10).withColumn(
                                               "L10_shots", sum('shots) over last10).withColumn(
                                               "L10_faceOffWins", sum('faceOffWins) over last10).withColumn(
                                               "L10_takeaways", sum('takeaways) over last10).withColumn(
                                               "L10_plusMinus", sum('plusMinus) over last10).withColumn(
                                               "L10_evenTimeOnIce", mean('evenTimeOnIce) over last10).withColumn(
                                               "L10_shortHandedTimeOnIce", mean('shortHandedTimeOnIce) over last10).withColumn(
                                               "L10_powerPlayTimeOnIce", mean('powerPlayTimeOnIce) over last10).withColumn(
                                               "l20_assists", sum('assists) over last20).withColumn(
                                               "L20_goals", sum('goals) over last20).withColumn(
                                               "L20_shots", sum('shots) over last20).withColumn(
                                               "L20_faceOffWins", sum('faceOffWins) over last20).withColumn(
                                               "L20_takeaways", sum('takeaways) over last20).withColumn(
                                               "L20_plusMinus", sum('plusMinus) over last20).withColumn(
                                               "L20_evenTimeOnIce", mean('evenTimeOnIce) over last20).withColumn(
                                               "L20_shortHandedTimeOnIce", mean('shortHandedTimeOnIce) over last20).withColumn(
                                               "L20_powerPlayTimeOnIce", mean('powerPlayTimeOnIce) over last20).orderBy(game("date_time").desc)
      //more feature age, points ,year 
      val lagdataseta = lagdataset.withColumn("age",datediff(col("date_time"),col("birthDate"))/365).withColumn(
                                              "points", col("goals") + col("assists"))withColumn(
                                               "year",year(col("date_time")))
      //Yearly stats with feature engeniring
      val dataByYearPlayers = lagdataseta.groupBy("player_id", "season").agg(countDistinct('game_id),last('team_id),first('team_id),mean('age),max('year),
          sum('assists),mean('assists),stddev_samp('assists),
          sum('goals),mean('goals),stddev_samp('goals),
          sum('points),mean('points),stddev_samp('points),
          sum('timeOnIce),mean('timeOnIce),stddev_samp('timeOnIce),
          sum('shots),mean('shots),stddev_samp('shots),
          sum('powerPlayGoals),mean('powerPlayGoals),stddev_samp('powerPlayGoals),
          sum('faceOffWins),mean('faceOffWins),stddev_samp('faceOffWins),
          sum('takeaways),mean('takeaways),stddev_samp('takeaways),
          sum('powerPlayTimeOnIce),mean('powerPlayTimeOnIce),stddev_samp('powerPlayTimeOnIce),
          last('l10_assists),last('L10_goals),last('L10_shots),last('L10_faceOffWins),last('L10_takeaways),last('L10_plusMinus),last('L10_evenTimeOnIce),
          last('L10_shortHandedTimeOnIce),last('L10_powerPlayTimeOnIce),last('l20_assists),last('L20_goals),last('L20_shots),last('L20_faceOffWins),last('L20_takeaways),
          last('L20_plusMinus),last('L20_evenTimeOnIce),last('L20_shortHandedTimeOnIce),last('L20_powerPlayTimeOnIce))
          
    
      
      //more feature rank and last 2 year stats
      val yrank = Window.partitionBy('season).orderBy(desc("sum(points)"))
      val dataByYearPlayer = dataByYearPlayers.withColumn("rank", row_number() over yrank)
      val nextyear= Window.partitionBy('player_id).orderBy('season).rowsBetween(1, 1)
      val allyear= Window.partitionBy('player_id).orderBy('season)
      val dataByYearPlayerNextt = dataByYearPlayer.withColumn("nexty_Team", sum("first(team_id, false)") over nextyear).withColumn(
                                               "nexty_points", sum("sum(points)") over nextyear).withColumn(
                                               "season_number", row_number() over allyear)                                                                              
      val dataByYearPlayerNexttt = dataByYearPlayerNextt.withColumn(
                                               "Lypointdif", (lag("avg(points)",1)) over allyear).withColumn(
                                               "Lygoaldif", (lag("avg(goals)",1))over allyear).withColumn(
                                               "Lyassistdif", (lag("avg(assists)",1))over allyear).withColumn(
                                               "L2ypointdif", (lag("avg(points)",2)) over allyear).withColumn(
                                               "L2ygoaldif", (lag("avg(goals)",2))over allyear).withColumn(
                                               "L2yassistdif", (lag("avg(assists)",2))over allyear)
                                           
      val dataByYearPlayerNext = dataByYearPlayerNexttt.withColumn(
                                               "difypointdif", col("avg(points)")- col("Lypointdif")).withColumn(
                                               "difygoaldif", col("avg(goals)") - col("Lygoaldif")).withColumn(
                                               "difyassistdif", col("avg(assists)") - col("Lyassistdif")).withColumn(
                                               "dif2ypointdif", col("avg(points)")- col("L2ypointdif")).withColumn(
                                               "dif2ygoaldif", col("avg(goals)") - col("L2ygoaldif")).withColumn(
                                               "dif2yassistdif", col("avg(assists)") - col("L2yassistdif")).withColumn(
                                               "Yearnum", col("max(year)")-2011)
   
 
      
      //create 3 Dataframe to train test and one to forecast this year.
      val traintmp = dataByYearPlayerNext.filter("nexty_Team >0").filter("rank<500")
      val tofroctmp = dataByYearPlayerNext.filter("season == '20182019'").filter("rank<500")
      val train =traintmp.na.fill(0)
      val tofroc =tofroctmp.na.fill(0)
      val colt = train.schema.names.clone()
      val colf = colt.filter(! _.contains("nexty_points"))
      val split = train.randomSplit(Array(0.8, 0.2), seed = 11L)
      val train_data = split(0).cache()
      val test_data = split(1).cache()
      
      // assembler and evaluator for regression 
      val assembler = new VectorAssembler().setInputCols(colf).setOutputCol("features")

      val evaluator = new RegressionEvaluator()
      .setLabelCol("nexty_points")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
      
      ///GenLinReg
      val glr = new GeneralizedLinearRegression()
        .setFamily("gaussian")
        .setLink("identity")
        .setMaxIter(8)
        .setRegParam(0.3)
        .setLabelCol("nexty_points")
        .setFeaturesCol("features")
      val pipelineglr = new Pipeline().setStages(Array(assembler, glr))
      val modelglr = pipelineglr.fit(train_data)
      val predictionsglr = modelglr.transform(test_data)
      predictionsglr.join(player, predictionsglr.col("player_id").equalTo(player("player_id"))).orderBy($"nexty_points".desc).select("prediction", "nexty_points", "firstName","lastName").show(50)
      val rmseglr = evaluator.evaluate(predictionsglr)
      println("Root Mean Squared Error (RMSE) on test data = " + rmseglr)  
      val predictions2020glr = modelglr.transform(tofroc)
      predictions2020glr.join(player, predictions2020glr.col("player_id").equalTo(player("player_id"))).orderBy($"prediction".desc).select("prediction","firstName","lastName").show(50)
     
      
      ///DecisionTreeRegressor
      val dtc = new RandomForestRegressor()
      .setLabelCol("nexty_points")
      .setFeaturesCol("features")
      .setMaxDepth(8)
     
    
      
      
      val pipeline = new Pipeline().setStages(Array(assembler, dtc))
      val model = pipeline.fit(train_data)
      val predictions = model.transform(test_data)
      predictions.join(player, predictions.col("player_id").equalTo(player("player_id"))).orderBy($"nexty_points".desc).select("prediction", "nexty_points", "firstName","lastName").show(50)
      val rmse = evaluator.evaluate(predictions)
      println("Root Mean Squared Error (RMSE) on test data = " + rmse)  
      val predictions2020 = model.transform(tofroc)
      predictions2020.join(player, predictions2020.col("player_id").equalTo(player("player_id"))).orderBy($"prediction".desc).select("prediction","firstName","lastName").show(50)
     
      ///GBTRegressor
      val gbt = new GBTRegressor()
      .setLabelCol("nexty_points")
      .setFeaturesCol("features")
      .setMaxIter(20)
      .setMaxDepth(6)
      .setMaxBins(150)
      .setStepSize(0.15)
      val evaluatorgbt = new RegressionEvaluator()
      .setLabelCol("nexty_points")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
      val pipelinegbt = new Pipeline().setStages(Array(assembler, gbt))

      
      ///Can use many param for cross validation but very very slow
      /*
      val paramGrid= new ParamGridBuilder()
                        .addGrid(gbt.maxDepth, 5 :: 10 :: 15 :: Nil)
                        .addGrid(gbt.maxIter,  5 :: 10 :: 15 :: Nil)
                       	.addGrid(gbt.maxBins, 5 :: 10 :: 15 :: Nil)
     										.addGrid(gbt.stepSize, .1 :: .2 :: Nil).build()   
                        
                  
      
      
      val numFolds = 2 
      
      
      val cv= new CrossValidator()
                  .setEstimator(pipelinegbt)
                  .setEvaluator(evaluatorgbt)
                  .setEstimatorParamMaps(paramGrid)
                  .setNumFolds(numFolds)
                  
      
      val modelgbt = cv.fit(train_data)
      */
      val modelgbt = pipelinegbt.fit(train_data)
      val predictionsgbt = modelgbt.transform(test_data)
      val predictionsgbt2020 = modelgbt.transform(tofroc)
      predictionsgbt.join(player, predictionsgbt.col("player_id").equalTo(player("player_id"))).orderBy($"nexty_points".desc).select("prediction", "nexty_points", "firstName","lastName").show(50)
      predictionsgbt2020.join(player, predictionsgbt2020.col("player_id").equalTo(player("player_id"))).orderBy($"prediction".desc).select("prediction","firstName","lastName").show(50)
      

      
      val rmsegbt = evaluatorgbt.evaluate(predictionsgbt)
      println(s"Root Mean Squared Error (RMSE) on test data = $rmsegbt")
/*
      val gbtModel = modelgbt.stages(1).asInstanceOf[GBTRegressionModel]
      println(s"Learned regression GBT model:\n ${gbtModel.toDebugString}")
      
      modelgbt.save("../NHLSPARK/src/NHLSPARKPACK/modelgbt");
      pipelinegbt.save("../NHLSPARK/src/NHLSPARKPACK/pipelinegbt");
     
      
      val pipelinegbt2 = Pipeline.load("../NHLSPARK/src/NHLSPARKPACK/pipelinegbt")
    
      val predictionsgbt20202 = modelgbt2.
      predictionsgbt.join(player, predictionsgbt.col("player_id").equalTo(player("player_id"))).orderBy($"nexty_points".desc).select("prediction", "nexty_points", "firstName","lastName").show(100)
      predictionsgbt2020.join(player, predictionsgbt2020.col("player_id").equalTo(player("player_id"))).orderBy($"prediction".desc).select("prediction","firstName","lastName").show(100)
      */
}