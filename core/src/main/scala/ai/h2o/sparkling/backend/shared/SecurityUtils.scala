/*
* Licensed to the Apache Software Foundation (ASF) under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The ASF licenses this file to You under the Apache License, Version 2.0
* (the "License"); you may not use this file except in compliance with
* the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package ai.h2o.sparkling.backend.shared

import org.apache.spark.h2o.H2OConf
import org.apache.spark.sql.SparkSession
import water.network.SparklingWaterSecurityUtils

private[backend] object SecurityUtils {

  def enableSSL(spark: SparkSession, conf: H2OConf): Unit = {
    val sslPair = SparklingWaterSecurityUtils.generateSSLPair()
    val config = SparklingWaterSecurityUtils.generateSSLConfig(sslPair)
    conf.set(SharedBackendConf.PROP_SSL_CONF._1, config)
    spark.sparkContext.addFile(sslPair.jks.getLocation)
    if (sslPair.jks.getLocation != sslPair.jts.getLocation) {
      spark.sparkContext.addFile(sslPair.jts.getLocation)
    }
  }

}
