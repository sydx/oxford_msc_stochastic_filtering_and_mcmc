import os

from pandas import DataFrame



class FileStructureElement(object):
    def verify(self):
        raise NotImplementedError('Pure virtual method')
    
    def updatecontext(self):
        raise NotImplementedError('Pure virtual method')

class DirectoryFileStructureElement(FileStructureElement):
    def __init__(self, path):
        self.__path = path
        
    @property
    def path(self): return self.__path
    
    def verify(self):
        return os.path.isdir(self.__path)
    
    def updatecontext(self, context, contextupdater):
        basename = os.path.basename(self.__path)
        contextupdater(context, basename)

class CsvFileStructureElement(FileStructureElement):
    def __init__(self, basename, fileobj):
        self.__basename = basename
        self.__fileobj = fileobj
    
    def verify(self):
        return not self.__fileobj.closed

class ZipFileStructureElement(FileStructureElement):
    def __init__(self, basename, fileobj):
        self.__basename = basename
        self.__fileobj = fileobj

    def verify(self):
        return not self.__fileobj.closed

class GzipFileStructureElement(FileStructureElement):
    def __init__(self, basename, fileobj):
        self.__basename = basename
        self.__fileobj = fileobj

    def verify(self):
        return not self.__fileobj.closed



class FileStructureLevel(object):
    def __init__(self, mayhavechildren, parentpath, regex):
        self.__mayhavechildren = mayhavechildren
        self.__parentpath = parentpath
        self.__regex = regex
        self.__children = []
        
    @property
    def parentpath(self): return self.__parentpath
        
    @property
    def regex(self): return self.__regex
    
    def matches(self, basename):
        return self.__regex.match(basename)
    
    def addchild(self, child):
        if not self.__mayhavechildren:
            raise NotImplementedError('This file structure level may not have children')
        self.__children.append(child)
    
    @property
    def haschildren(self):
        return self.__children == True
    
    @property
    def hasdata(self): raise NotImplementedError('Pure virtual property')
    
    def elements(self): raise NotImplementedError('Pure virtual property')
    
class DirectoryFileStructureLevel(FileStructureLevel):
    def __init__(self, parentpath, regex):
        super(DirectoryFileStructureLevel, self).__init__(True, parentpath, regex)
    
    @property
    def hasdata(self): return False
    
    def elements(self):
        elements = []
        basenames = os.listdir(self.parentpath)
        for basename in basenames:
            if self.matches(basename):
                path = os.path.join(self.parentpath, basename)
                if os.path.isdir(path):
                    elements.append(DirectoryFileStructureElement(path))
        return elements

class CsvFileStructureLevel(FileStructureLevel):
    def __init__(self, parentpath, regex):
        super(DirectoryFileStructureLevel, self).__init__(False, parentpath, regex)
    
    @property
    def hasdata(self): return True
    
    def elements(self):
        elements

class ZipFileStructureLevel(FileStructureLevel):
    def __init__(self, parentpath, regex):
        super(DirectoryFileStructureLevel, self).__init__(True, parentpath, regex)
    
    @property
    def hasdata(self): return False

class GzipFileStructureLevel(FileStructureLevel):
    def __init__(self, parentpath, regex):
        super(DirectoryFileStructureLevel, self).__init__(True, parentpath, regex)
    
    @property
    def hasdata(self): return False



class FileStructure(object):
    def __init__(self, rootlevel):
        self.__rootlevel = rootlevel
        
    def __loaddata(self, level, df, context):
        elements = level.elements()
        if level.hasdata:
            for element in elements:
                # Load data from el
                pass
        if level.haschildren:
            for element in elements:
                for elementchild in element.children:
                    for levelchild in level.children:
                        
            
    def loaddata(self):
        context = {}
        df = DataFrame()
        