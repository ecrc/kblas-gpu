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
                sh "module load gcc/4.8.5 cuda/7.0; make clean; make all" 
            }
        }
        stage ('cuda-7.5') {
            steps {
                sh "module load gcc/4.8.5 cuda/7.5; make clean; make all" 
            }
        }
        stage ('cuda-8.0') {
            steps {
                sh "module load gcc/4.8.5 cuda/8.0; make clean; make all" 
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

}
