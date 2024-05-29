import os
import shutil
from subprocess import call

'''
display('log')
check('C:/Users/Paul Bilokon/Documents/dev/alexandria/bilokon-msc/dissertation/code/mcmc/svl.txt')
data('C:/temp/2/dataset-14-256-1_n.txt')
data('C:/temp/2/dataset-14-256-1_y.txt')
compile(1)
inits(1, 'C:/Users/Paul Bilokon/Documents/dev/alexandria/bilokon-msc/dissertation/code/mcmc/svl-inits.txt')
gen.inits()
update(1000)
set(beta)
set(mu)
set(phi)
set(rho)
set(sigmav)
update(1000)
coda(*, 'C:/temp/2/coda-')
stats(*, 'C:/temp/2/stats')
save('C:/temp/2/out.txt')
quit()
'''

ticksperday = 128

#datasetdirpath = r"C:\Users\Paul Bilokon\Documents\dev\alexandria\bilokon-msc\dissertation\code\datasets\dataset-14"
datasetdirpath = r"C:\Users\Paul\Documents\dev\alexandria\bilokon-msc\dissertation\code\datasets\dataset-14"
outdirpath = 'C:/temp/3'

for i in range(101,111):
    filebasename = 'dataset-14-%d-%d' % (ticksperday, i)
    tempoutdirpath = os.path.join('c:/temp', filebasename).replace('\\', '/')
    os.mkdir(tempoutdirpath)
    nfilepath = os.path.join(datasetdirpath, str(ticksperday), '%s_n.txt' % filebasename).replace('\\', '/')
    yfilepath = os.path.join(datasetdirpath, str(ticksperday), '%s_y.txt' % filebasename).replace('\\', '/')
    shutil.copy(nfilepath, tempoutdirpath)
    shutil.copy(yfilepath, tempoutdirpath)
    scriptfilepath = os.path.join(tempoutdirpath, 'script.txt').replace('\\', '/')
    with open(scriptfilepath, 'w') as o:
        o.write("display('log')")
        o.write("check('C:/Users/Paul/Documents/dev/alexandria/bilokon-msc/dissertation/code/mcmc/svl2.txt')\n")
        o.write("data('%s')\n" % os.path.join(tempoutdirpath, '%s_n.txt' % filebasename).replace('\\', '/'))
        o.write("data('%s')\n" % os.path.join(tempoutdirpath, '%s_y.txt' % filebasename).replace('\\', '/'))
        o.write("compile(1)\n")
        o.write("inits(1, 'C:/Users/Paul/Documents/dev/alexandria/bilokon-msc/dissertation/code/mcmc/svl-inits.txt')\n")
        o.write("gen.inits()\n")
        o.write("update(10000)\n")
        o.write("set(beta)\n")
        o.write("set(mu)\n")
        o.write("set(phi)\n")
        o.write("set(rho)\n")
        o.write("set(sigmav)\n")
        o.write("update(100000)\n")
        o.write("coda(*, '%s/coda')\n" % tempoutdirpath)
        o.write("stats(*)\n")
        o.write("save('%s/out.txt')\n" % tempoutdirpath)
        o.write("quit()\n")
    call(r'"C:\Program Files\WinBUGS14\WinBUGS14.exe" /PAR "%s" /HEADLESS' % scriptfilepath)
    #call(r'"C:\Program Files (x86)\OpenBUGS\OpenBUGS323\OpenBUGS.exe" /PAR "%s" /HEADLESS' % scriptfilepath)
    
    os.mkdir(os.path.join(outdirpath, filebasename))
    nodestatisticsfilepath = os.path.join(outdirpath, filebasename, 'node-statistics.txt')

    with open(os.path.join(tempoutdirpath, 'out.txt')) as o:
        for line in o:
            if line == 'Node statistics\n':
                with open(nodestatisticsfilepath, 'w') as nsf:
                    for line in o:
                        if not line.startswith('\t'): break
                        nsf.write(line)
    shutil.copy(os.path.join(tempoutdirpath, 'coda1.txt'), os.path.join(outdirpath, filebasename, 'coda-chain-1.txt'))
    shutil.copy(os.path.join(tempoutdirpath, 'codaIndex.txt'), os.path.join(outdirpath, filebasename, 'coda-index.txt'))
