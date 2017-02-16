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
          '--range 512:15360:512  ',        #square matrices
          #'--mrange 512:15360:512 -n 512 ', #tall & skinny matrices
          #'--nrange 512:15360:512 -m 512 '  #thin & wide matrices
          ]

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

    for p in programs:
        sys.stdout.write('running: '+p+' ... ')
        if (not os.path.isfile(BIN_PATH+p)):
            print 'Unable to find '+BIN_PATH+p+' executable! Skipping...'
        else:
            os.system('echo running: '+p+' > '+TEST_LOGS_PATH+'/'+p+'.txt')
            for v in variants:
                for r in ranges:
                    cmd = (BIN_PATH+p+' '+r+' -w --nb 128 --db 256 -t '+check+' --dev 0 '+v)
                    os.system('echo >> '+TEST_LOGS_PATH+'/'+p+'.txt')
                    os.system('echo '+cmd+' >> '+TEST_LOGS_PATH+'/'+p+'.txt')
                    sys.stdout.flush()
                    os.system(cmd+' >> '+TEST_LOGS_PATH+'/'+p+'.txt')
                    time.sleep(1)
            print 'done'

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

    for p in programs:
        sys.stdout.write('running: '+p+' ... ')
        if (not os.path.isfile(BIN_PATH+p)):
            print 'Unable to find '+BIN_PATH+p+' executable! Skipping...'
        else:
            os.system('echo running: '+p+' > '+TEST_LOGS_PATH+'/'+p+'.txt')
            for v in variants:
                for r in ranges:
                    cmd = (BIN_PATH+p+' '+r+' -w --nb 128 --db 256 -t '+check+' --dev 0 '+v)
                    os.system('echo >> '+TEST_LOGS_PATH+'/'+p+'.txt')
                    os.system('echo '+cmd+' >> '+TEST_LOGS_PATH+'/'+p+'.txt')
                    sys.stdout.flush()
                    os.system(cmd+' >> '+TEST_LOGS_PATH+'/'+p+'.txt')
                    time.sleep(1)
            print 'done'

