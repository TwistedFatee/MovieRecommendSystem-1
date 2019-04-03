package com.zhangchao.statistics

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String, shoot: String, language: String, genres: String, actors: String, directors: String)

case class Rating(uid: Int, mid: Int, score: Double, timestamp: Int)

case class MongoConfig(uri:String, db:String)

case class Recommendation(mid:Int, score:Double)

case class GenresRecommendation(genres:String, recs:Seq[Recommendation])



object StatisticsRecommender {

  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_MOVIE_COLLECTION = "Movie"

  //统计的表的名称
  val RATE_MORE_MOVIES = "RateMoreMovies"
  val RATE_MORE_RECENTLY_MOVIES = "RateMoreRecentlyMovies"
  val AVERAGE_MOVIES = "AverageMovies"
  val GENRES_TOP_MOVIES = "GenresTopMovies"



  // 入口方法
  def main(args: Array[String]): Unit = {

    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://hadoop102:27017/recommender",
      "mongo.db" -> "recommender"
    )
    val conf: SparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("DataLoader")
    //ssc
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    import spark.implicits._

    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    //从mongoDB加载数据
    val ratingDF: DataFrame = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Rating]
      .toDF()

    val movieDF: DataFrame = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie]
      .toDF()

      //创建临时表
      ratingDF.createOrReplaceTempView("ratings")
    // 1. 历史热门统计，历史评分数据最多，mid，count
    val rateMoreMoviesDF: DataFrame = spark.sql("select mid,count(mid)  as count from ratings group by mid order by count desc")
    //将数据保存到mongo数据库中
    storeDFInMongoDB( rateMoreMoviesDF, RATE_MORE_MOVIES )

    // 2. 近期热门统计，按照“yyyyMM”格式选取最近的评分数据，统计评分个数
    val simpleDateFormat = new SimpleDateFormat("yyyyMM")
    spark.udf.register("changeDate", (x:Int)=>simpleDateFormat.format(new Date(x*1000L)).toInt)
    //对原始数据做处理,去掉UID
    val fraratingOfYearMonthme: DataFrame = spark.sql("select mid,score,changeDate(timestamp) as yearmonth from ratings")
    fraratingOfYearMonthme.createOrReplaceTempView("ratingOfMonth")

    // 从ratingOfMonth中查找电影在各个月份的评分，mid，count，yearmonth
    val rateMoreRecentlyMoviesDF: DataFrame = spark.sql("select mid,count(mid),yearmonth from ratingOfMonth group by yearmonth,mid order by yearmonth desc ,mid desc")

    //存入DB数据库中
    storeDFInMongoDB(rateMoreRecentlyMoviesDF,RATE_MORE_RECENTLY_MOVIES)

    // 3. 优质电影统计，统计电影的平均评分，mid，avg
    val averageMoviesDF: DataFrame = spark.sql("select mid,avg(score) avg from ratings group by mid")
      //存入数据库中
      storeDFInMongoDB(averageMoviesDF,AVERAGE_MOVIES)

    //4.统计每种电影类型中评分最高的10个电影
      //定义所有类别
      val genres = List("Action","Adventure","Animation","Comedy","Crime","Documentary","Drama","Family","Fantasy","Foreign","History","Horror","Music","Mystery"
        ,"Romance","Science","Tv","Thriller","War","Western")
      //把平均分加入movie表
      val movieWithScore: DataFrame = movieDF.join(averageMoviesDF,"mid")
      //将genres转为RDD
    val genresRDD: RDD[String] = spark.sparkContext.makeRDD(genres)
      //计算类别top10，
    val genresTopMoviesDF: DataFrame = genresRDD.cartesian(movieWithScore.rdd)
      .filter {
        case (genres, movierow) => movierow.getAs[String]("genres").toLowerCase.contains(genres.toLowerCase())
      }
      .map {
        case (genre, movieRow) => (genre, (movieRow.getAs[Int]("mid"), movieRow.getAs[Double]("avg")))
      }
      .groupByKey()
      .map {
        case (genre, movieRow) => GenresRecommendation(genre, movieRow.toList.sortWith(_._2 > _._2).take(10).map(item => Recommendation(item._1, item._2)))
      }.toDF()

    //保存数据
    storeDFInMongoDB(genresTopMoviesDF,GENRES_TOP_MOVIES)

    //ssc.sql("select ")


  }
  def storeDFInMongoDB(df: DataFrame, collection_name: String)(implicit mongoConfig: MongoConfig): Unit = {

    df.write
      .option("uri", mongoConfig.uri)
      .option("collection", collection_name)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

  }

}