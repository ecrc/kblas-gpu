pipeline {
/*
 * Defining where to run
 */
//// Any:
// agent any
//// By agent label:
//      agent { label 'sandybridge' }

    agent { label 'Almaha' }
    triggers {
        pollSCM('H/10 * * * *')
    }
    environment {
        CC="gcc"
    }

    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }

    stages {
        stage ('cuda-8.0') {
            steps {
                sh '''#!/bin/bash -le
                    module load gcc/4.8.5;
                    module load cuda/8.0
                    module load intel/16
                    module list
                    set -x
                    export _MAGMA_ROOT_=/opt/ecrc/magma/2.2.0-intel-16-mkl-cuda-8.0
                    export _CUB_DIR_=$PWD/cub
                    if [ -d cub ]
                    then
                        cd cub; git pull; cd ..
                    else
                        git clone https://github.com/NVLABS/cub cub
                    fi
                    make clean
                    make
                    export CUDA_VISIBLE_DEVICES=2; export NGPUS=1
                    sed -i s/STEP_DIM=.*/STEP_DIM=1024/ ./test-scripts/kblas-test-l2.sh
                    sed -i s/STOP_DIM=.*/STOP_DIM=4096/ ./test-scripts/kblas-test-l2.sh
                    ./test-scripts/kblas-test-l2.sh
                    sed -i s/"ranges = "/"ranges=\\[\\"--range 128:1024:128\\"\\]\\nranges = "/ ./test-scripts/kblas-test-l3.py
                    sed -i "/ranges = /,/\\]/d" ./kblas-test-l3.py
                    ./test-scripts/kblas-test-l3.py
                    sed -i "/--range 2048:15360:1024/d" ./test-scripts/kblas-test-l3.py
                    ./test-scripts/kblas-test-l3.py
                    ./test-scripts/kblas-test-batch-parallel.py
                '''
            }
        }
        stage ('cuda-7.5') {
            steps {
                sh '''#!/bin/bash -le
                    module load gcc/4.8.5;
                    module load cuda/7.5
                    module load intel/16
                    module list
                    set -x
                    export _MAGMA_ROOT_=/opt/ecrc/magma/2.2.0-intel-16-mkl-cuda-7.5
                    export _CUB_DIR_=$PWD/cub
                    if [ -d cub ]
                    then
                        cd cub; git pull; cd ..
                    else
                        git clone https://github.com/NVLABS/cub cub
                    fi
                    make clean
                    make
                    export CUDA_VISIBLE_DEVICES=2; export NGPUS=1
                    sed -i s/STEP_DIM=.*/STEP_DIM=1024/ ./test-scripts/kblas-test-l2.sh
                    sed -i s/STOP_DIM=.*/STOP_DIM=4096/ ./test-scripts/kblas-test-l2.sh
                    ./test-scripts/kblas-test-l2.sh
                    sed -i s/"ranges = "/"ranges=\\[\\"--range 128:1024:128\\"\\]\\nranges = "/ ./test-scripts/kblas-test-l3.py
                    sed -i "/ranges = /,/\\]/d" ./test-scripts/kblas-test-l3.py
                    ./test-scripts/kblas-test-l3.py
                    sed -i "/--range 2048:15360:1024/d" ./test-scripts/kblas-test-l3.py
                    ./test-scripts/kblas-test-l3.py
                    ./test-scripts/kblas-test-batch-parallel.py
                '''
            }
        }
        stage ('cuda-7.0') {
            steps {
                sh '''#!/bin/bash -le
                    module load gcc/4.8.5;
                    module load cuda/7.0
                    module load intel/16
                    module list
                    set -x
                    export _MAGMA_ROOT_=/opt/ecrc/magma/2.0.1-intel-16-mkl-cuda-7.0/
                    export _CUB_DIR_=$PWD/cub
                    if [ -d cub ]
                    then
                        cd cub; git pull; cd ..
                    else
                        git clone https://github.com/NVLABS/cub cub
                    fi
                    make clean
                    make
                    export CUDA_VISIBLE_DEVICES=2; export NGPUS=1
                    sed -i s/STEP_DIM=.*/STEP_DIM=1024/ ./test-scripts/kblas-test-l2.sh
                    sed -i s/STOP_DIM=.*/STOP_DIM=4096/ ./test-scripts/kblas-test-l2.sh
                    ./test-scripts/kblas-test-l2.sh
                    sed -i s/"ranges = "/"ranges=\\[\\"--range 128:1024:128\\"\\]\\nranges = "/ ./test-scripts/kblas-test-l3.py
                    sed -i "/ranges = /,/\\]/d" ./test-scripts/kblas-test-l3.py
                    ./test-scripts/kblas-test-l3.py
                    sed -i "/--range 2048:15360:1024/d" ./test-scripts/kblas-test-l3.py
                    ./test-scripts/kblas-test-l3.py
                    ./test-scripts/kblas-test-batch-parallel.py
                '''
            }
        }
    }
    // Post build actions
    post {
        //always {
        //}
        //success {
        //}
        //unstable {
        //}
        //failure {
        //}
        unstable {
                emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build is UNSTABLE", recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'RequesterRecipientProvider']]
        }
        failure {
                emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build FAILED", recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'RequesterRecipientProvider']]
        }
    }
}
