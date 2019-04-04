package com.atguigu.offline

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.jblas.DoubleMatrix

/**
  * @Project OfflineRecommender
  * @Auther: zhangchao
  * @Date: 2019/4/3 16:18
  * @Description:
  */

// 基于评分数据的LFM，只需要rating数据
case class MovieRating(uid: Int, mid: Int, score: Double, timestamp: Int )

case class MongoConfig(uri:String, db:String)

// 定义一个基准推荐对象
case class Recommendation( mid: Int, score: Double )

// 定义基于预测评分的用户推荐列表
case class UserRecs( uid: Int, recs: Seq[Recommendation] )

// 定义基于LFM电影特征向量的电影相似度列表
case class MovieRecs( mid: Int, recs: Seq[Recommendation] )

object OfflineRecommender {


  // 定义表名和常量
  val MONGODB_RATING_COLLECTION = "Rating"

  val USER_RECS = "UserRecs"
  val MOVIE_RECS = "MovieRecs"

  val USER_MAX_RECOMMENDATION = 20

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://Hadoop102:27017/recommender",
      "mongo.db" -> "recommender"
    )
    //spark实例
    val sparkConf: SparkConf = new SparkConf().setAppName("OfflineRecommender").setMaster(config("spark.cores"))
    val spark: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()

    implicit val mongoConfig=MongoConfig(config("mongo.uri" ),config("mongo.db"))
    import spark.implicits._
    //从数据库中获取评分数据
    val ratingRDD: RDD[(Int, Int, Double)] = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map(rating => (rating.uid, rating.mid, rating.score))
      .cache()

    //从ratingRDD获取所有的mid,uid,并且去重
    val userRDD: RDD[Int] = ratingRDD.map(_._1).distinct()
    val movieRDD: RDD[Int] = ratingRDD.map(_._2).distinct()

    //将ratingRDD转换成训练隐语义模型参数
    val trainData: RDD[Rating] = ratingRDD.map(x=>Rating(x._1,x._2,x._3))

    val (rank,iter,lambda)=(200,5,0.1)
    val model: MatrixFactorizationModel = ALS.train(trainData,rank,iter,lambda)

    //基于用户和电影的隐形特征，计算预测评分，得到用户的推荐列表（uid，list（mid,预测评分））

    //计算user和movie的笛卡尔积，得到一个预测空矩阵
    val userMovie: RDD[(Int, Int)] = userRDD.cartesian(movieRDD)

    //调用模型predicate预测评分
    val preRatings: RDD[Rating] = model.predict(userMovie)

    //对预测数据进行处理
    val userRecs: DataFrame = preRatings.filter(_.rating > 0) //过滤出评分大于0的项
      .map(rating => (rating.user, (rating.product, rating.rating)))
      .groupByKey()
      .map {
        case (uid, recs) => UserRecs(uid, recs.toList.sortWith(_._2 > _._2).take(USER_MAX_RECOMMENDATION).map(x => Recommendation(x._1, x._2)))
      }
      .toDF()

   //将推荐数据传送到DB中
    userRecs.write
      .option("uri",mongoConfig.uri)
      .option("collection",USER_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //基于电影隐特征，计算相似度矩阵，得到电影的相似度列表（mid,list(mid,相似度)）
    val moveFeatures: RDD[(Int, DoubleMatrix)] = model.productFeatures.map {
      case (mid, features) => (mid, new DoubleMatrix(features))
    }
    val movieRecs: DataFrame = moveFeatures.cartesian(moveFeatures)
      .filter {
        //自己跟自己配对的过滤掉
        case (a, b) => a._1 != b._2
      }
      .map {
        case (a, b) => {
          val simScore: Double = this.consinSim(a._2, b._2)

          (a._1, (b._1, simScore))
        }
      } //过滤出相似度大于0.6的
      .filter(_._2._2 > 0.6)
      .groupByKey()
      .map {
        //转换成样例类并将相似度排序
        case (mid, item) => MovieRecs(mid, item.toList.sortWith(_._2 > _._2).map(x => Recommendation(x._1, x._2)))
      }
      .toDF()


    //将电影相似度保存到DB中
    movieRecs.write
      .option("uri",mongoConfig.uri)
      .option("collection",MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
    

  }
  //求向量余弦相似度
  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2)/(movie1.norm2()*movie2.norm2())

  }


}
