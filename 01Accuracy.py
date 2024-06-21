import numpy
from sklearn import metrics
actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)
Precision = metrics.precision_score(actual, predicted)
print(Precision)

import numpy
from sklearn import metrics
actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)
Sensitivity_recall = metrics.recall_score(actual, predicted)
print(Sensitivity_recall)