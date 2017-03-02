#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, getopt
import csv
import time
import commands

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

    
cmd=('nvidia-smi -L | wc -l')
NGPUS = commands.getstatusoutput(cmd)[1]
if (NGPUS < '1'):
    print 'Unable to detect an NVIDIA GPU device to test on! Exiting'
    exit()

#check = ''
check = '-c'
TRMM = 1
TRSM = 1 
ranges = ['--range 128:1024:128',           #square matrices
          '--range 2048:15360:1024 ',        #square matrices
          #'--mrange 512:15360:512 -n 512 ', #tall & skinny matrices
          #'--nrange 512:15360:512 -m 512 '  #thin & wide matrices
          ]
#--------------------------------
def task1(pVariants, pRanges, pExec, pCheck, pDev, pOutfile, pGpus):
    sys.stdout.write('running: '+pExec+' ... ')
    os.system('echo running: '+pExec+' > '+pOutfile)
    for v in pVariants:
        for r in pRanges:
            cmd = (pExec+' '+r+' -w --nb 128 --db 256 --ngpu '+pGpus+' -t '+pCheck+' '+v)
            os.system('echo >> '+pOutfile)
            os.system('echo '+cmd+' >> '+pOutfile)
            sys.stdout.flush()
            os.system(cmd+' >> '+pOutfile)
            time.sleep(1)
    print ' done'

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
    programs = ['test_strmm', 'test_dtrmm', 'test_ctrmm', 'test_ztrmm',
                'test_strmm_cpu', 'test_dtrmm_cpu', 'test_ctrmm_cpu', 'test_ztrmm_cpu',
                'test_strmm_mgpu', 'test_dtrmm_mgpu', 'test_ctrmm_mgpu', 'test_ztrmm_mgpu'
               ]

    for p in programs:
        pp = BIN_PATH+p
        if (not os.path.isfile(pp)):
            print 'Unable to find '+pp+' executable! Skipping...'
        else:
            logFile = TEST_LOGS_PATH+'/'+p+'.txt'
            task1(variants, ranges, pp, check, 0, logFile, NGPUS)

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
    programs = ['test_strsm', 'test_dtrsm', 'test_ctrsm', 'test_ztrsm',
                'test_strsm_cpu', 'test_dtrsm_cpu', 'test_ctrsm_cpu', 'test_ztrsm_cpu',
                'test_strsm_mgpu', 'test_dtrsm_mgpu', 'test_ctrsm_mgpu', 'test_ztrsm_mgpu'
                ]

    for p in programs:
        pp = BIN_PATH+p
        if (not os.path.isfile(pp)):
            print 'Unable to find '+pp+' executable! Skipping...'
        else:
            logFile = TEST_LOGS_PATH+'/'+p+'.txt'
            task1(variants, ranges, pp, check, 0, logFile, NGPUS)


