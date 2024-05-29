import csv
from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

datasets = [14, 15]
models = ['svl', 'svl2']
powers = 8

meanrhos = {}

for dataset in datasets:
    for model in models:
        meanrhos[(dataset, model)] = []

        fig = plt.figure()
        plot = fig.add_subplot(111)

        for po2 in range(0, powers):
            ticksperday = 2**po2
            datadir = '%d' % ticksperday
            datetimesfile = os.path.join(datadir, 'datetimes.txt')
            datetimes = []
            with open(datetimesfile, 'r') as dtf:
                reader = csv.reader(dtf)
                for line in reader:
                    format = '%Y-%m-%d %H:%M:%S.%f'
                    print(datetime.strptime(line[0], format), datetime.strptime(line[1], format))
                    datetimes.append((datetime.strptime(line[0], format), datetime.strptime(line[1], format)))

            rhos = []

            for i, (dt1, dt2) in enumerate(datetimes):
                resultsdir = os.path.join('..', '..', 'results', 'mcmc', model, 'dataset-%d-%d-%d' % (dataset, ticksperday, i+1))
                nodestatisticspath = os.path.join(resultsdir, 'node-statistics.txt')
                if os.path.exists(nodestatisticspath):
                    with open(nodestatisticspath) as nsf:
                        reader = csv.reader(nsf, delimiter='\t')
                        for line in reader:
                            if line[1] == 'rho':
                                rho = float(line[2])
                                rhos.append(rho)
                    print(resultsdir)
                    plot.plot([dt1, dt2], [rho, rho], 'b', linewidth=3*(powers-po2), alpha=1.*(0.75**po2))

            meanrhos[(dataset, model)].append(np.mean(rhos))

        plot.axhline(y=0., color='r')
        plot.set_xlabel('time', fontsize=12)
        plot.set_ylabel(r'$\hat{\rho}_{\tau}$', fontsize=12)
        plot.tick_params(axis='both', which='major', labelsize=9)
        plot.tick_params(axis='both', which='minor', labelsize=9)
        plot.set_title('Dataset %s, %s' % (dataset, model.upper()), fontsize=12)

fig = plt.figure()
plot = fig.add_subplot(111)
for dataset in datasets:
    for model in models:
        plot.plot(range(0, powers), meanrhos[(dataset, model)])
plot.tick_params(axis='both', which='major', labelsize=9)
plot.tick_params(axis='both', which='minor', labelsize=9)

fig = plt.figure()
plot = fig.add_subplot(111)
timeintervalsindays = 1. / 2.**np.array(range(0, powers))
lntimeintervalsindays = np.log(timeintervalsindays)
for dataset in datasets:
    for model in models:
        plot.plot(lntimeintervalsindays, meanrhos[(dataset, model)], 'o--', label='Dataset %s, %s' % (dataset, model.upper()))
        print((dataset, model))
        print(scipy.stats.linregress(x=lntimeintervalsindays, y=meanrhos[(dataset, model)]))
plot.set_xlabel(r'$\ln(\tau)$', fontsize=12)
plot.set_ylabel(r'$\hat{\rho}_{\tau}$', fontsize=12)
plot.tick_params(axis='both', which='major', labelsize=9)
plot.tick_params(axis='both', which='minor', labelsize=9)
plot.legend(loc='upper right', fancybox=True, framealpha=0.5, prop={'size': 8})

plt.show()

