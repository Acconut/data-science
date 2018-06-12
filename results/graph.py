import numpy as np
import mglearn
from matplotlib import pyplot

scores = np.array([[0.28539862, 0.85884745, 0.88713099, 0.88518649, 0.86512286],
       [0.36211773, 0.85009722, 0.88721937, 0.89075482, 0.87590596],
       [0.06080962, 0.8020152 , 0.86096871, 0.87316599, 0.86335514],
       [0.05683224, 0.80537387, 0.87095634, 0.87776206, 0.87493371]])

mglearn.tools.heatmap(
	scores,
	xlabel='learning_rate', xticklabels=[1, 0.1, 0.01, 0.001, 0.0001],
	ylabel='hidden_layer_sizes', yticklabels=['1 x 50', '1 x 100', '2 x 50', '2 x 100'],
	fmt='%0.3f')

pyplot.savefig('../paper/fig2.pdf')

