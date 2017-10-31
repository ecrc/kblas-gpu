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
        stage ('cuda-7.0') {
            steps {
                sh '''#!/bin/bash -le
                    module load gcc/4.8.5 cuda/7.0; make clean; make all
                    export CUDA_VISIBLE_DEVICES=0; export NGPUS=1
                    sed -i s/STEP_DIM=.*/STEP_DIM=1024/ ./kblas-test-l2.sh
                    sed -i s/STOP_DIM=.*/STOP_DIM=4096/ ./kblas-test-l2.sh
                    ./kblas-test-l2.sh
                    sed -i s/"ranges = "/"ranges=\\[\\"--range 128:1024:128\\"\\]\\nranges = "/ ./kblas-test-l3.py
                    sed -i "/ranges = /,/\\]/d" ./kblas-test-l3.py
                    ./kblas-test-l3.py
                    sed -i "/--range 2048:15360:1024/d" ./kblas-test-l3.py
                    ./kblas-test-l3.py
                '''
            }
        }
        stage ('cuda-7.5') {
            steps {
                sh '''#!/bin/bash -le
                    module load gcc/4.8.5 cuda/7.5; make clean; make all
                    export CUDA_VISIBLE_DEVICES=0; export NGPUS=1
                    sed -i s/STEP_DIM=.*/STEP_DIM=1024/ ./kblas-test-l2.sh
                    sed -i s/STOP_DIM=.*/STOP_DIM=4096/ ./kblas-test-l2.sh
                    ./kblas-test-l2.sh
                    sed -i s/"ranges = "/"ranges=\\[\\"--range 128:1024:128\\"\\]\\nranges = "/ ./kblas-test-l3.py
                    sed -i "/ranges = /,/\\]/d" ./kblas-test-l3.py
                    ./kblas-test-l3.py
                    sed -i "/--range 2048:15360:1024/d" ./kblas-test-l3.py
                    ./kblas-test-l3.py
                '''
            }
        }
        stage ('cuda-8.0') {
            steps {
                sh '''#!/bin/bash -le
                    module load gcc/4.8.5 cuda/8.0; make clean; make all
                    export CUDA_VISIBLE_DEVICES=0; export NGPUS=1
                    sed -i s/STEP_DIM=.*/STEP_DIM=1024/ ./kblas-test-l2.sh
                    sed -i s/STOP_DIM=.*/STOP_DIM=4096/ ./kblas-test-l2.sh
                    ./kblas-test-l2.sh
                    sed -i s/"ranges = "/"ranges=\\[\\"--range 128:1024:128\\"\\]\\nranges = "/ ./kblas-test-l3.py
                    sed -i "/ranges = /,/\\]/d" ./kblas-test-l3.py
                    ./kblas-test-l3.py
                    sed -i "/--range 2048:15360:1024/d" ./kblas-test-l3.py
                    ./kblas-test-l3.py
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
