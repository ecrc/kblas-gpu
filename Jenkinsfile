pipeline {
    agent { label 'Kanary' }
    triggers {
        pollSCM('H/10 * * * *')
    }
    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }
    stages {
        stage ('build') {
            steps {
            sh '''#!/bin/bash -el
                    # The -x flags indicates to echo all commands, thus knowing exactly what is being executed.
                    # The -e flags indicates to halt on error, so no more processing of this script will be done
                    echo $(pwd)
                    module purge
                    module load gcc/10.2.0
        		    module load openblas/0.3.18-gcc-10.2.0
		            export OPENBLAS_ROOT=/opt/local/openblas/0.3.18-gcc-10.2.0
		            export LD_LIBRARY_PATH=/opt/local/openblas/0.3.18-gcc-10.2.0/lib:/opt/rocm/hipblas/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
		            make 
'''
    }
           }
               
        stage ('test') {
            steps {
            sh '''#!/bin/bash -el
                    echo $(pwd)
                    module purge
		    module load gcc/10.2.0
		    module load openblas/0.3.18-gcc-10.2.0
                    export OPENBLAS_ROOT=/opt/local/openblas/0.3.18-gcc-10.2.0
                    export LD_LIBRARY_PATH=/opt/local/openblas/0.3.18-gcc-10.2.0/lib:/opt/rocm/hipblas/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
		    echo $(pwd)
		    source testing/testscript.sh
'''
}
    }
        }
          }
