#!/bin/bash

OIFS="$IFS"
IFS=$'\n'

rm dataset_ddr/*

cd "dataset_ddr/packs"

if [ -d "../stepcharts" ]; then
	rm -rf "../stepcharts"
fi
mkdir "../stepcharts"

if [ -d "../audiofiles" ]; then
	rm -rf "../audiofiles"
fi
mkdir "../audiofiles"

if [ -d "../train" ]; then
	rm -rf "../train"
fi
mkdir "../train"

if [ -d "../test" ]; then
	rm -rf "../test"
fi
mkdir "../test"

if [ -d "../val" ]; then
	rm -rf "../val"
fi
mkdir "../val"

for f in $(find -name "*.sm"); do
	echo "Copying $f ..."
	cp "$f" "../stepcharts"
done
#echo $(ls -l ../stepcharts | wc -l) "files copied"

for f in $(find -name "*.mp3" -o -name "*.wav"); do
	echo "Converting $f to ogg"
	ffmpeg -hide_banner -loglevel quiet -n -i $f "${f%.*}.ogg"
	rm $f
done

for f in $(find -name "*.ogg"); do
	echo "Copying $f ..."
	cp "$f" "../audiofiles"
done

cd ../..

python parser.py

rm -rf dataset_ddr/audiofiles/
rm -rf dataset_ddr/stepcharts/

./create_index.sh ./dataset_ddr
cd dataset_ddr
python ../split.py
rm -rf index.txt
cd ..

./create_index.sh dataset_ddr/train
./create_index.sh dataset_ddr/val
./create_index.sh dataset_ddr/test

IFS="$OIFS"
