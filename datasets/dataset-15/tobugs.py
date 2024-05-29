import csv
import os

import numpy as np

filename = 'dataset-15-256.csv'
basename = filename.split('.')[0]
ticksperday = int(basename.split('-')[-1])
os.mkdir(str(ticksperday))

count = 0
datetimes = []
prices = []
with open(filename, 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for row in reader:
        datetimes.append(row[0])
        prices.append(float(row[1]))
        count += 1

print(count, ticksperday)
print(count / ticksperday)

datasetsize = count / ticksperday

with open('%d/datetimes.txt' % ticksperday, 'w') as fd:
    for i in range(ticksperday):
        print('dataset %d:' % i)
        print(datasetsize*i, datasetsize*(i+1))
        dt = datetimes[datasetsize*i : datasetsize*(i+1)]
        ps = prices[datasetsize*i : datasetsize*(i+1)]
        lr = [np.log(ps[j]) - np.log(ps[j-1]) for j in xrange(1, len(ps))]
        lr -= np.average(lr)
        lr *= 100.
        lrc = 0
        with open('%d/dataset-14-%d-%d_y.txt' % (ticksperday, ticksperday, i+1), 'w') as fy:
            fy.write('y[]\n')
            for x in lr:
                fy.write('%f\n' % x)
                lrc += 1
            fy.write('END\n')
        with open('%d/dataset-14-%d-%d_n.txt' % (ticksperday, ticksperday, i+1), 'w') as fn:
            fn.write('list(n=%d)\n' % lrc)
        fd.write('%s,%s\n' % (dt[0], dt[-1]))
        print(np.average(lr))
