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
NGPUS=0
# first check using environment variable
if ( "CUDA_VISIBLE_DEVICES" in os.environ ):
    aux=os.environ["CUDA_VISIBLE_DEVICES"].strip(',')
    if ( len(aux) > 0):
      NGPUS=int( aux.count(',') ) + 1
if ( NGPUS == 0 ):
    # check using system
    cmd=('nvidia-smi -L | wc -l')
    NGPUS = int(commands.getstatusoutput(cmd)[1])

if (NGPUS < 1):
    print 'Unable to detect an NVIDIA GPU device to test on! Exiting'
    exit()

print 'NGPUS: ' + str(NGPUS)

#set options
#check = ''
check = ' -c'
defaultBatchCount = 100
TEST_BATCH_SVD = 1
TEST_BATCH_QR = 1
TEST_BATCH_TRSM = 1
TEST_BATCH_TRMM = 1
TEST_BATCH_GEMM = 1
TEST_BATCH_SYRK = 1
TEST_BATCH_POTRF = 1
TEST_BATCH_LAUUM = 1
TEST_BATCH_TRTRI = 1
TEST_BATCH_POTRS = 1
TEST_BATCH_POTRI = 1

#--------------------------------
def task1(pVariants, pRanges, pExec, pOptions, pBatchCount, pDev, pOutfile):
    print 'running: '+pExec+' ... '
    os.system('echo running: '+pExec+' > '+pOutfile)
    for v in pVariants:
        for r in pRanges:
            cmd = (pExec+' '+ r + ' ' + pOptions + ' --dev ' + str(pDev) + ' ' + v + ' --batchCount ' + str(pBatchCount))
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

#--------------------------------
def parallelTaskLaunch(variants, programs, ranges, options, batchCount):
    deviceQueue = [[] for _ in xrange(NGPUS)]
    deviceThread = []

    dev = 0
    for p in programs:
        pp = BIN_PATH+p
        if (not os.path.isfile(pp)):
            print 'Unable to find '+pp+' executable! Skipping...'
        else:
            logFile = TEST_LOGS_PATH+'/'+p+'.txt'
            deviceQueue[dev].append( threading.Thread(target=task1, args=(variants, ranges, pp, options, batchCount, dev, logFile)) )
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

############### BATCH_SVD
if (TEST_BATCH_SVD == 1):
    variants = ['']
    programs = ['test_dgesvj_batch', 'test_sgesvj_batch']
    ranges = ['--range 32:512:32']
    options = ''
    batchCount = 200;

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)

############### BATCH_QR
if (TEST_BATCH_QR == 1):
    variants = ['']
    programs = ['test_dgeqrf_batch', 'test_sgeqrf_batch']
    ranges = ['--range 32:512:32']
    batchCount = defaultBatchCount;
    options = ''

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)

############### BATCH_GEMM
if (TEST_BATCH_GEMM == 1):
    variants = ['-NN',
                '-TN',
                '-NT',
                '-TT',
                ]
    programs = ['test_sgemm_batch',
                'test_dgemm_batch',
                'test_cgemm_batch',
                'test_zgemm_batch'
                ]
    ranges = [
              '--range 2:70+1',
              '--range 32:256:32'
              ]
    options = check
    batchCount = defaultBatchCount;

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)

############### BATCH_TRMM
if (TEST_BATCH_TRMM == 1):
    variants = ['-L -NN -DN',
                '-L -TN -DN',
                # '-SL -U -NN -DN',
                # '-SL -U -TN -DN',
                # '-SR -L -NN -DN',
                # '-SR -L -TN -DN',
                # '-SR -U -NN -DN',
                # '-SR -U -TN -DN'
                ]

    programs = ['test_strmm_batch',
                'test_dtrmm_batch',
                'test_ctrmm_batch',
                'test_ztrmm_batch'
                ]
    ranges = [
              '--range 2:70+1',
              '--range 32:256:32'
              ]
    options = check
    batchCount = defaultBatchCount;

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)

############### BATCH_TRSM
if (TEST_BATCH_TRSM == 1):
    variants = ['-SL -L -NN -DN',
                '-SL -L -TN -DN',
                # '-SL -U -NN -DN',
                # '-SL -U -TN -DN',
                '-SR -L -NN -DN',
                '-SR -L -TN -DN',
                # '-SR -U -NN -DN',
                # '-SR -U -TN -DN'
                ]
    programs = ['test_strsm_batch',
                'test_dtrsm_batch',
                'test_ctrsm_batch',
                'test_ztrsm_batch'
                ]
    ranges = [
              '--range 2:70+1',
              '--range 32:256:32'
              ]
    options = check
    batchCount = defaultBatchCount;

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)

############### BATCH_SYRK
if (TEST_BATCH_SYRK == 1):
    variants = ['-L -NN',
                '-L -TN',
                # '-U -NN',
                # '-U -TN',
                ]
    programs = ['test_ssyrk_batch',
                'test_dsyrk_batch',
                'test_csyrk_batch',
                'test_zsyrk_batch'
                ]
    ranges = [
              '--range 2:70+1',
              '--range 32:256:32'
              ]
    options = check
    batchCount = defaultBatchCount;

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)

############### BATCH_POTRF
if (TEST_BATCH_POTRF == 1):
    variants = ['-L',
                # '-U',
                ]
    programs = ['test_spotrf_batch',
                'test_dpotrf_batch',
                'test_cpotrf_batch',
                'test_zpotrf_batch'
                ]
    ranges = [
              '--range 2:70+1',
              '--range 32:256:32'
              ]
    options = check
    batchCount = defaultBatchCount;

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)

############### BATCH_LAUUM
if (TEST_BATCH_LAUUM == 1):
    variants = ['-L',
                # '-U',
                ]
    programs = ['test_slauum_batch',
                'test_dlauum_batch',
                'test_clauum_batch',
                'test_zlauum_batch'
                ]
    ranges = [
              '--range 2:70+1',
              '--range 32:256:32'
              ]
    options = check
    batchCount = defaultBatchCount;

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)

############### BATCH_TRTRI
if (TEST_BATCH_TRTRI == 1):
    variants = ['-L -DN',
                # '-L -DU',
                # '-U -DN',
                # '-U -DU',
                ]
    programs = ['test_strtri_batch',
                'test_dtrtri_batch',
                'test_ctrtri_batch',
                'test_ztrtri_batch'
                ]
    ranges = [
              '--range 2:70+1',
              '--range 32:256:32'
              ]
    options = check
    batchCount = defaultBatchCount;

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)

############### BATCH_TRTRI
if (TEST_BATCH_POTRS == 1):
    variants = ['-SR -L',
                # '-SR -U',
                # '-SL -L',
                # '-SL -U',
                ]
    programs = ['test_spotrs_batch',
                'test_dpotrs_batch',
                'test_cpotrs_batch',
                'test_zpotrs_batch'
                ]
    ranges = [
              '--range 2:70+1',
              '--range 32:256:32'
              ]
    options = check
    batchCount = defaultBatchCount;

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)

############### BATCH_POTRI
if (TEST_BATCH_POTRI == 1):
    variants = ['-L',
                # '-U',
                ]
    programs = ['test_spotri_batch',
                'test_dpotri_batch',
                'test_cpotri_batch',
                'test_zpotri_batch'
                ]
    ranges = [
              '--range 2:70+1',
              '--range 32:256:32'
              ]
    options = check
    batchCount = defaultBatchCount;

    parallelTaskLaunch(variants, programs, ranges, options, batchCount)
