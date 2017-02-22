#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, getopt
import csv
import time
import commands
import threading

# create output log folder

KBLAS_HOME='.'
TEST_LOGS_PATH='kblas-test-log'
cmd = ('cd '+KBLAS_HOME+'; mkdir -p '+TEST_LOGS_PATH)
print cmd
sys.stdout.flush()
os.system(cmd)

BIN_PATH='./testing/bin/'
if (not os.path.isdir(BIN_PATH)):
    print 'Unable to find executables folder! Exiting'
    exit()

#detect GPU devices
cmd=('nvidia-smi -L | wc -l')
NGPUS = int(commands.getstatusoutput(cmd)[1])
if (NGPUS < 1):
    print 'Unable to detect an NVIDIA GPU device to test on! Exiting'
    exit()

#set options
#check = ''
check = '-c'
TRMM = 1
TRSM = 1
ranges = ['--range 128:1024:128',           #square matrices
          '--range 2048:15360:1024  ',        #square matrices
          #'--mrange 512:15360:512 -n 512 ', #tall & skinny matrices
          #'--nrange 512:15360:512 -m 512 '  #thin & wide matrices
          ]

          
#--------------------------------
def task1(pVariants, pRanges, pExec, pCheck, pDev, pOutfile):
    print 'running: '+pExec+' ... '
    os.system('echo running: '+pExec+' > '+pOutfile)
    for v in pVariants:
        for r in pRanges:
            cmd = (pExec+' '+r+' -w --nb 128 --db 256 -t '+pCheck+' --dev '+str(pDev)+' '+v)
            os.system('echo >> '+pOutfile)
            os.system('echo '+cmd+' >> '+pOutfile)
            sys.stdout.flush()
            os.system(cmd+' >> '+pOutfile)
            time.sleep(1)
    print pExec+' done'

#--------------------------------
def launchQueue(*taskQueue):
    for t in taskQueue:
        t.start()
        t.join()




############### TRMM
if (TRMM == 1):
    variants = ['-SL -L -NN',
                '-SL -L -TN',
                '-SL -U -NN',
                '-SL -U -TN',
                '-SR -L -NN',
                '-SR -L -TN',
                '-SR -U -NN',
                '-SR -U -TN'
                ]
    programs = ['test_strmm', 'test_dtrmm', 'test_ctrmm', 'test_ztrmm', 'test_strmm_cpu', 'test_dtrmm_cpu', 'test_ctrmm_cpu', 'test_ztrmm_cpu']

    deviceQueue = [[] for _ in xrange(NGPUS)]
    deviceThread = []
    
    dev = 0
    for p in programs:
        pp = BIN_PATH+p
        if (not os.path.isfile(pp)):
            print 'Unable to find '+pp+' executable! Skipping...'
        else:
            logFile = TEST_LOGS_PATH+'/'+p+'.txt'
            deviceQueue[dev].append( threading.Thread(target=task1, args=(variants, ranges, pp, check, dev, logFile)) )
            dev = (dev+1)%NGPUS

    dev = 0
    while dev < NGPUS:
        q = deviceQueue[dev]
        #print q
        tq = threading.Thread( target=launchQueue, args=(q) )
        deviceThread.append(tq)
        tq.start()
        dev = dev + 1

    dev = 0
    while dev < NGPUS:
        tq = deviceThread[dev]
        tq.join()
        dev = dev + 1


############### TRSM
if (TRSM == 1):
    variants = ['-SL -L -NN',
                '-SL -L -TN',
                '-SL -U -NN',
                '-SL -U -TN',
                '-SR -L -NN',
                '-SR -L -TN',
                '-SR -U -NN',
                '-SR -U -TN'
                ]
    programs = ['test_strsm', 'test_dtrsm', 'test_ctrsm', 'test_ztrsm', 'test_strsm_cpu', 'test_dtrsm_cpu', 'test_ctrsm_cpu', 'test_ztrsm_cpu']

    deviceQueue = [[] for _ in xrange(NGPUS)]
    deviceThread = []

    dev = 0
    for p in programs:
        pp = BIN_PATH+p
        if (not os.path.isfile(pp)):
            print 'Unable to find '+pp+' executable! Skipping...'
        else:
            logFile = TEST_LOGS_PATH+'/'+p+'.txt'
            deviceQueue[dev].append( threading.Thread(target=task1, args=(variants, ranges, pp, check, dev, logFile)) )
            dev = (dev+1)%NGPUS

    dev = 0
    while dev < NGPUS:
        q = deviceQueue[dev]
        #print q
        tq = threading.Thread( target=launchQueue, args=(q) )
        deviceThread.append(tq)
        tq.start()
        dev = dev + 1

    dev = 0
    while dev < NGPUS:
        tq = deviceThread[dev]
        tq.join()
        dev = dev + 1

