#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, getopt
import csv
import time

check = '' #'-c'
TRMM = 1
TRSM = 0
SYMM = 0
SYRK = 0
ranges = [#'--range 128:1024:128',           #square matrices
          '--range 512:15360:512  ',        #square matrices
          '--mrange 512:15360:512 -n 512 ', #tall & skinny matrices
          '--nrange 512:15360:512 -m 512 '  #thin & wide matrices
          ]

if (TRMM == 1):
    variants = ['-SL -L -NN',
                '-SL -L -TN',
                '-SL -U -NN',
                '-SL -U -TN',
                '-SR -L -NN',
                '-SR -L -TN',
                '-SR -U -NN',
                '-SR -U -TN']
    programs = ['test_strmm', 'test_dtrmm', 'test_strmm_cpu', 'test_dtrmm_cpu']

    for v in variants:
        for r in ranges:
            for p in programs:
                print
                cmd = ('./testing/'+p+' '+r+' -w --nb 128 --db 256 -t '+check+' --dev 1 '+v)
                print cmd
                sys.stdout.flush()
                os.system(cmd)
                time.sleep(2)

############### TRSM
if (TRSM == 1):
    variants = ['-SL -L -NN',
                '-SL -L -TN',
                '-SL -U -NN',
                '-SL -U -TN',
                '-SR -L -NN',
                '-SR -L -TN',
                '-SR -U -NN',
                '-SR -U -TN']
    programs = ['test_strsm', 'test_dtrsm', 'test_strsm_cpu', 'test_dtrsm_cpu']

    for v in variants:
        for r in ranges:
            for p in programs:
                print
                cmd = ('./testing/'+p+' '+r+' -w --nb 128 --db 256 -t '+check+' --dev 4 '+v)
                print cmd
                sys.stdout.flush()
                os.system(cmd)
                time.sleep(2)

############### SYMM
if (SYMM == 1):
	print
	cmd = ('./testing/test_dsymm_lap  '+range+' -w -SL -L')
	print cmd
	sys.stdout.flush()
	os.system(cmd)
	time.sleep(2)
	
	print
	cmd = ('./testing/test_dsymm_lap  '+range+' -w -SL -U')
	print cmd
	sys.stdout.flush()
	os.system(cmd)
	time.sleep(2)
	
	print
	cmd = ('./testing/test_dsymm_lap  '+range+' -w -SR -L')
	print cmd
	sys.stdout.flush()
	os.system(cmd)
	time.sleep(2)
	
	print
	cmd = ('./testing/test_dsymm_lap  '+range+' -w -SR -U')
	print cmd
	sys.stdout.flush()
	os.system(cmd)
	time.sleep(2)

if (SYRK == 1):
	print
	cmd = ('./testing/test_dsyrk_cpu  '+range+' -w -NN -L')
	print cmd
	sys.stdout.flush()
	os.system(cmd)
	time.sleep(2)
	
	print
	cmd = ('./testing/test_dsyrk_cpu  '+range+' -w -NN -U')
	print cmd
	sys.stdout.flush()
	os.system(cmd)
	time.sleep(2)
	
	print
	cmd = ('./testing/test_dsyrk_cpu  '+range+' -w -TN -L')
	print cmd
	sys.stdout.flush()
	os.system(cmd)
	time.sleep(2)

	print
	cmd = ('./testing/test_dsyrk_cpu  '+range+' -w -TN -U')
	print cmd
	sys.stdout.flush()
	os.system(cmd)
	time.sleep(2)
