import numpy as np

learn_start = 50000
lr_start = 0.00005
lr_end = 0.00001
lr_endt = 500000

for numSteps in range(0, 1000000, 10000):
    t = max(0, numSteps - learn_start)
    lr = (lr_start - lr_end) * (lr_endt - t) / lr_endt + lr_end
    lr = max(lr, lr_end)
    print str(numSteps)+"\t"+format(lr, "0.6f")
