import math

class FlatStoredArray(object):
    def __init__(self, *args):
        self.__count = self._getcount(*args)
        self._data = [None] * self.__count
    
    def _getcount(self):
        raise NotImplementedError('Pure virtual method')
    
    def _keytoindex(self, key):
        raise NotImplementedError('Pure virtual method')
    
    def _indextokey(self, index):
        raise NotImplementedError('Pure virtual method')

    def __getitem__(self, key):
        return self._data[self._keytoindex(key)]
    
    def __setitem__(self, key, value):
        self._data[self._keytoindex(key)] = value
        
    def __len__(self):
        return self.__count
    
    def __repr__(self):
        return repr(self._data)
    
    def __str__(self):
        return str(self._data)
    
    def setall(self, iterable):
        for i, v in enumerate(iterable):
            if i >= self.__count: break
            self._data[i] = v
            
    class __Iterator(object):
        def __init__(self, data):
            self._data = data
            self.__idx = 0
            
        def __iter__(self):
            return self
            
        def __next__(self):
            if self.__idx < len(self._data):
                v = self._data[self.__idx]
                self.__idx += 1
                return v
            raise StopIteration()

    def __iter__(self):
        return FlatStoredArray.__Iterator(self._data)
    
    class __KeysIterator(object):
        def __init__(self, collection):
            self.__collection = collection
            self.__idx = 0
            
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.__idx < len(self.__collection):
                k = self.__collection._indextokey(self.__idx)
                self.__idx += 1
                return k
            raise StopIteration()
        
    def keys(self):
        return FlatStoredArray.__KeysIterator(self)

    class __ItemsIterator(object):
        def __init__(self, data, collection):
            self.__data = data
            self.__collection = collection
            self.__idx = 0
            
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.__idx < len(self.__data):
                k = self.__collection._indextokey(self.__idx)
                v = self.__data[self.__idx]
                self.__idx += 1
                return k, v
            raise StopIteration()
        
    def items(self):
        return FlatStoredArray.__ItemsIterator(self._data, self)
    
class DiagonalArray(FlatStoredArray):
    def __init__(self, dim):
        super(DiagonalArray, self).__init__(dim)
        self.__dim = dim
        
    @property
    def dim(self): return self.__dim
    
    @classmethod
    def _getcount(cls, dim):
        return (dim*dim + dim) // 2
    
    @classmethod
    def _keytoindex(cls, key):
        i, j = key[0], key[1]
        if i < j: i, j = j, i
        return (i*i + i) // 2 + j
    
    @classmethod
    def _indextokey(self, index):
        i = int(math.sqrt(2*index))
        n = (i*i + i) // 2
        j = index - n
        if j < 0:
            i -= 1
            n = (i*i + i) // 2
            j = index - n
        return i, j
    
    @classmethod
    def mindim(cls, count):
        dim = int(math.sqrt(2*count))
        if cls._getcount(dim) < count:
            dim += 1
        return dim
    
    @classmethod
    def create(cls, obj):
        if isinstance(obj, DiagonalArray):
            res = DiagonalArray(obj.dim)
            res.setall(obj)
        elif isinstance(obj, SubdiagonalArray):
            res = DiagonalArray(obj.dim)
            for k, v in obj.items():
                self[k] = v
        else:
            res = DiagonalArray(cls.mindim(len(obj)))
            res.setall(obj)
        return res
    
    def tonumpyarray(self, fill=None, symmetric=False):
        import numpy as np
        if fill is None: fill = np.NAN
        res = np.empty((self.__dim, self.__dim))
        idx = 0
        for i in range(self.__dim):
            for j in range(i+1):
                res[i,j] = self._data[idx]
                if symmetric: res[j,i] = res[i,j]
                idx += 1
            if not symmetric: res[i,i+1:self.__dim] = fill
        return res
        
class SubdiagonalArray(FlatStoredArray):
    def __init__(self, dim):
        super(SubdiagonalArray, self).__init__(dim)
        self.__dim = dim
        
    @property
    def dim(self): return self.__dim
    
    @classmethod
    def _getcount(cls, dim):
        return (dim*dim - dim) // 2
        
    @classmethod
    def _keytoindex(cls, key):
        i, j = key[0], key[1]
        if i < j: i, j = j, i
        return (i*i - i) // 2 + j

    @classmethod
    def _indextokey(cls, index):
        i = int(math.sqrt(2*index)) + 1
        n = (i*i - i) // 2
        j = index - n
        if j < 0:
            i -= 1
            n = (i*i - i) // 2
            j = index - n
        return i, j
    
    @classmethod
    def mindim(cls, count):
        dim = int(math.sqrt(2*count)) + 1
        if cls._getcount(dim) < count:
            dim += 1
        return dim
    
    @classmethod
    def create(cls, obj):
        if isinstance(obj, SubdiagonalArray):
            res = SubdiagonalArray(obj.dim)
            res.setall(obj)
        elif isinstance(obj, DiagonalArray):
            res = SubdiagonalArray(obj.dim)
            for k, v in obj.items():
                if k[0] != k[1]: self[k] = v
        else:
            res = SubdiagonalArray(cls.mindim(len(obj)))
            res.setall(obj)
        return res

    def tonumpyarray(self, fill=None, symmetric=False):
        import numpy as np
        if fill is None: fill = np.NAN
        res = np.empty((self.__dim, self.__dim))
        idx = 0
        for i in range(self.__dim):
            for j in range(i):
                res[i,j] = self._data[idx]
                if symmetric: res[j,i] = res[i,j]
                idx += 1
            res[i,i] = fill
            if not symmetric: res[i,i+1:self.__dim] = fill
        return res
    