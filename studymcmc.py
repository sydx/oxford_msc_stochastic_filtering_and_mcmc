import csv

def readindexfile(indexfilepath):
    index = {}
    with open(indexfilepath, 'rt') as indexfile:
        reader = csv.reader(indexfile, delimiter='\t')
        for row in reader:
            varname, startindex, endindex = row
            index[varname] = (startindex, endindex)
    return index    

def main():
    indexfilepath = r'C:\Users\Paul Bilokon\Documents\dev\alexandria\bilokon-msc\dissertation\code\winbugs\svl2\dataset-1\coda-index.txt'
    chainfilepath = r'C:\Users\Paul Bilokon\Documents\dev\alexandria\bilokon-msc\dissertation\code\winbugs\svl2\dataset-1\coda-for-chain-1.txt'
    
    index = readindexfile(indexfilepath)
    print(index)
    
    data = []
    indexoffset = None
    with open(chainfilepath, 'rt') as chainfile:
        reader = csv.reader(chainfile, delimiter='\t')
        for row in reader:
            index, value = int(row[0]), float(row[1])
            if not data: indexoffset = index
            print(index, indexoffset)
            assert index == len(data) + indexoffset
            data.append(value)
    print(data)

if __name__ == '__main__':
    main()
