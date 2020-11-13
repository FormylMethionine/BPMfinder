cd dataset_ddr

if [ -d "./train" ]; then
	rm -rf "./train"
fi
mkdir "./train"

if [ -d "./test" ]; then
	rm -rf "./test"
fi
mkdir "./test"

if [ -d "./val" ]; then
	rm -rf "./val"
fi
mkdir "./val"

python ../split.py

../create_index.sh train
../create_index.sh test
../create_index.sh val
