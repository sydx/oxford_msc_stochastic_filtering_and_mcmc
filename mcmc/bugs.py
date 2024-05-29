from collections import OrderedDict
import csv
from six import string_types

import numpy as np
import statsmodels.api as sm

class BUGSChain(object):
	def __init__(self, indexfile, datafile):
		self.__indexfile = indexfile
		self.__datafile = datafile
		self.__nodenames = []
		self.__nodedata = None
		
	def load(self):
		if isinstance(self.__indexfile, string_types):
			self.__indexfile = open(self.__indexfile, 'r')
		if isinstance(self.__datafile, string_types):
			self.__datafile = open(self.__datafile, 'r')
		try:
			indexreader = csv.reader(self.__indexfile, delimiter='\t')
			nodeindices = OrderedDict()
			observationcount = None
			for row in indexreader:
				start = int(row[1]) - 1
				end = int(row[2])
				nodeindices[row[0]] = (start, end)
				if observationcount is None:
					observationcount = end - start
				else:
					assert observationcount == end - start
					
			if observationcount is None: return
			
			datareader = csv.reader(self.__datafile, delimiter='\t')
			indices, data = [], []
			for row in datareader:
				indices.append(int(row[0]))
				data.append(float(row[1]))
			
			nodecount = len(nodeindices)
			self.__nodedata = np.empty((observationcount, nodecount)) 
				
			for i, (name, (start, end)) in enumerate(nodeindices.items()):
				self.__nodenames.append(name)
				self.__nodedata[:,i] = data[start:end]
		finally:
			self.__indexfile.close()
			self.__datafile.close()
			
	def kde(self, nodenames=None):
		if nodenames is None: nodenames = self.__nodenames
		nodeindices = [self.__nodenames.index(nn) for nn in nodenames]
		return sm.nonparametric.KDEMultivariate(
				data=self.__nodedata[:,nodeindices],
				var_type=('c' * len(nodeindices)))
