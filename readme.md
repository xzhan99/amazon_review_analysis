## Instructions
### Contents
    /stage1
        run_stage1.sh
        script_stage1.py
    /stage2
        run_stage2.sh
        script_stage2.py
    /stage3
        run_stage3.sh
        script_stage3.py
    /stage4
        run_stage4.sh
        script_stage4.py
    bootstrapping.sh
    cluster_config.json
    README.md
    requirements.txt
### Environmental settings
The project is designed to run on EMR cluster.It is developed and tested on the release emr-5.23.0. The environmental settings may be different for each release, make sure you use the right version of EMR cluster before running.<br>
The EMR cluster should contains the following softwares: Hadoop, Spark, Livy, TensorFLow.<br>
File 'bootstrapping.sh' and 'cluster_config.json' are given in the repository for software configurations and installations. These two files should be used at stage 1 and stage 3 respectively when you create the cluster.

### How to run
The project has a clear and straightforward content structure. All useful files for each stage are under independent folders.<br>
In general, each stage has one shell file named 'run_stagex.sh' and one python file 'script_stagex.py'. You can start each stage by running its own shell script.<br>
To run stage 1 codes, you should execute the following command:

    sh run_stage1.sh [input_location]
To run stage 2 codes, you should execute the following command:

    sh run_stage2.sh [input_location] [output_location]
To run stage 3 codes, you should execute the following command:

    sh run_stage3.sh [input_location]
To run stage 4 codes, you should execute the following command:

    sh run_stage4.sh [input_location]
Notice, the input and output location is considered as a HDFS path by default. If either input or output file is on the local file system, the command should be like:
    
    sh run_stage1.sh file:///home/hadoop/amazon_reviews_us_Music_v1_00.tsv
