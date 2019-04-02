package com.zhangchao.recommerder

import com.mongodb.casbah.commons.MongoDBObject
import com.mongodb.casbah.{MongoClient, MongoClientURI}
import org.apache.spark.SparkConf
import org.apache.spark.internal.config
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}


//样例类
case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String,
                 shoot: String, language: String, genres: String, actors: String,
                 directors: String)
case class Rating(uid: Int, mid: Int, score: Double, timestamp: Int)
case class Tag(uid: Int, mid: Int, tag: String, timestamp: Int)

case class MongoConfig(uri:String, db:String)
case class ESConfig(httpHosts:String, transportHosts:String, index:String,
                    clustername:String)

object DataLoader {

    val MOVEIES_PATH="C:\\Users\\zhangchao\\IdeaProjects\\MovieRecommendSystem3\\recommender\\DataLoader\\src\\main\\resources\\movies.csv"

  val RATINGS_PATH="C:\\Users\\zhangchao\\IdeaProjects\\MovieRecommendSystem3\\recommender\\DataLoader\\src\\main\\resources\\ratings.csv"

  val TAGS_PATH="C:\\Users\\zhangchao\\IdeaProjects\\MovieRecommendSystem3\\recommender\\DataLoader\\src\\main\\resources\\tags.csv"
  val MONGODB_MOVIE_COLLECTION = "Movie"
  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_TAG_COLLECTION = "Tag"
  val ES_MOVIE_INDEX = "Movie"



  def main(args: Array[String]): Unit = {
    //conf
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://hadoop102:27017/recommender",
      "mongo.db" -> "recommender",
      "es.httpHosts" -> "hadoop102:9200",
      "es.transportHosts" -> "hadoop102:9300",
      "es.index" -> "recommender",
      "es.cluster.name" -> "elasticsearch"
    )
    val conf: SparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("DataLoader")
    //ssc
    val ssc: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    //电影
    val movieRDD: RDD[String] = ssc.sparkContext.textFile(MOVEIES_PATH)
      //将RDD转换成df
    import ssc.implicits._
    val movieDF: DataFrame = movieRDD.map(row => {
      val files: Array[String] = row.split("\\^")
      Movie(files(0).toInt, files(1).trim, files(2).trim, files(3).trim, files(4).trim, files(5).trim, files(6).trim, files(7).trim, files(8).trim, files(9).trim)
    }).toDF()

    val ratingRDD = ssc.sparkContext.textFile(RATINGS_PATH)

    val ratingDF = ratingRDD.map(item => {
      val attr = item.split(",")
      Rating(attr(0).toInt,attr(1).toInt,attr(2).toDouble,attr(3).toInt)
    }).toDF()

    val tagRDD = ssc.sparkContext.textFile(TAGS_PATH)
    //将tagRDD装换为DataFrame
    val tagDF = tagRDD.map(item => {
      val attr = item.split(",")
      Tag(attr(0).toInt,attr(1).toInt,attr(2).trim,attr(3).toInt)
    }).toDF()

    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))
    //将文件写进MOngodb中
    storeDataInMongoDB(movieDF,ratingDF,tagDF)

    ssc.stop()
  }


  def storeDataInMongoDB(movieDF: DataFrame, ratingDF:DataFrame , tagDF: DataFrame)(implicit mongoConfig: MongoConfig): Unit = {
    // 新建一个mongodb的连接
    val mongoClient = MongoClient(MongoClientURI(mongoConfig.uri))
    // 如果mongodb中已经有相应的数据库，先删除
    mongoClient(mongoConfig.db)(MONGODB_MOVIE_COLLECTION).dropCollection()
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).dropCollection()
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).dropCollection()

    // 将DF数据写入对应的mongodb表中
    movieDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    ratingDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    tagDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_TAG_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //对数据表建索引
    mongoClient(mongoConfig.db)(MONGODB_MOVIE_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).createIndex(MongoDBObject("uid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).createIndex(MongoDBObject("uid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).createIndex(MongoDBObject("mid" -> 1))

    mongoClient.close()


  }

}
