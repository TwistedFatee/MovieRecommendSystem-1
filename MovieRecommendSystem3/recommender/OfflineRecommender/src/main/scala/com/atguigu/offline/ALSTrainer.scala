package com.atguigu.offline

import breeze.numerics.sqrt
import com.atguigu.offline.OfflineRecommender.MONGODB_RATING_COLLECTION
import org.apache.avro.TestDataFileSpecific
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
  * @Project ALSTrainer
  * @Auther: zhangchao
  * @Date: 2019/4/3 18:54
  * @Description:
  */
object ALSTrainer {


  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://hadoop102:27017/recommender",
      "mongo.db" -> "recommender"
    )

    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("ALSTrainer")

    //创建spark
    val spark: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    //读取数据库中的数据
    val ratingRDD: RDD[Rating] = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map(rating => Rating(rating.uid, rating.mid, rating.score)) // 转化成rdd，并且去掉时间戳
      .cache()


    //随机切分数据集，生成训练集和测试集
    val splits = ratingRDD.randomSplit(Array(0.8, 0.2))
    val trainningRDD = splits(0)
    val testRDD = splits(1)

    //模型参数选择，输出最有参数
    adjustALSParam(trainningRDD, ratingRDD)

    spark.close()

  }


  def adjustALSParam(trainningRDD: RDD[Rating], testRDD: RDD[Rating]) = {
    val result: Array[(Int, Double, Double)] = for (rank <- Array(50,100,200,300) ;lambda <- Array(0.01,0.1,1))
      yield {
        val model: MatrixFactorizationModel = ALS.train(trainningRDD,rank,5,lambda)

        //计算当前参数的对应模型的均方根误差，返回doulbe
        val rmse: Double = getRMSE(model,testRDD)
        (rank,lambda,rmse)
      }
    //打出最优模型参数
    println(result.minBy(_._3))

  }
  def getRMSE(model: MatrixFactorizationModel, data: RDD[Rating]): Double = {
    //计算空的模型
    val userProducts: RDD[(Int, Int)] = data.map(item=>(item.user,item.product))
    val predictRting: RDD[Rating] = model.predict(userProducts)

    //以uid,mid作为外键关联两个RDD，观测实际值与预测值
      //将实际值转换成键值对
    val observed: RDD[((Int, Int), Double)] = data.map{item=>((item.user,item.product),item.rating)}

      //将预测值编程键值对
    val predict: RDD[((Int, Int), Double)] = predictRting.map(item=>((item.user,item.product),item.rating))

      //内连接并对数据进行处理

    sqrt{
      observed.join(predict).map{
        case ((uid,mid),(act,pre))=>
          val err: Double = act-pre
          err *err
      }.mean()
    }
  }
}
