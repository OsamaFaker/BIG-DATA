from pyspark import SparkConf, SparkContext
import collections
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions
import numpy as np
from pyspark.sql.functions import lit
from pyspark.sql import SQLContext
import numpy as np
from pyspark.ml.linalg import Vectors
import time
print('---------------------------------------------------Start')
TrD = np.load('TrD.npy')
TrL = np.load('TrLI.npy')
TsD = np.load('TsD.npy')
TsL = np.load('TsLI.npy')
print('------------------------------------------------------Data Loaded')
Tr=np.hstack((TrL.reshape(-1,1),TrD))
Ts=np.hstack((TsL.reshape(-1,1),TsD))
conf = SparkConf()
sc = SparkContext(conf = conf)
spark = SQLContext(sc)
dff = map(lambda x: (int(x[0]), Vectors.dense(x[1:])), Tr)
TrDF = spark.createDataFrame(dff,schema=["label", "features"])
dff = map(lambda x: (int(x[0]), Vectors.dense(x[1:])), Ts)
TsDF = spark.createDataFrame(dff,schema=["label", "features"])
smr=[]
print('-------------------------------------------------------Dataframes Created')
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
layers = [43, 32, 10]
trainer = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=1000000, seed=1234)
st = time.time()
model = trainer.fit(TrDF)
tet = time.time()-st
result = model.transform(TsDF)
st = time.time()
result = model.transform(TsDF)
pet = time.time()-st
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
acc = evaluator.evaluate(predictionAndLabels)
cm = np.array(result.select("prediction", "label").collect())
np.savetxt('CM1.csv',cm,delimiter=',',fmt='%s')
smr.append([1, tet, pet, acc])
np.savetxt('Summary1.csv', np.array(smr),delimiter=',',fmt='%s')
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
layers = [43, 64, 32, 10]
trainer = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=1000000, seed=1234)
st = time.time()
model = trainer.fit(TrDF)
tet = time.time()-st
st = time.time()
result = model.transform(TsDF)
pet = time.time()-st
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
acc = evaluator.evaluate(predictionAndLabels)
cm = np.array(result.select("prediction", "label").collect())
np.savetxt('CM2.csv',cm,delimiter=',',fmt='%s')
smr.append([2, tet, pet, acc])
np.savetxt('Summary2.csv', np.array(smr),delimiter=',',fmt='%s')
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
layers = [43, 128, 64, 32, 10]
trainer = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=1000000, seed=1234)
st = time.time()
model = trainer.fit(TrDF)
tet = time.time()-st
st = time.time()
result = model.transform(TsDF)
pet = time.time()-st
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
acc = evaluator.evaluate(predictionAndLabels)
cm = np.array(result.select("prediction", "label").collect())
np.savetxt('CM3.csv',cm,delimiter=',',fmt='%s')
smr.append([3, tet, pet, acc])
np.savetxt('Summary.csv', np.array(smr),delimiter=',',fmt='%s')
