TEST_DATA_DIR=./data/test/query
mkdir -p $TEST_DATA_DIR
aws s3 cp s3://drivendata-competition-meta-vsc-data-us/test/query $TEST_DATA_DIR --recursive --region us-east-1 --no-sign-request --quiet