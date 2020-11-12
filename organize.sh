#!/bin/bash

OIFS="$IFS"
IFS=$'\n'

cd "dataset_ddr/packs"

if [ -d "../stepcharts" ]; then
	rm -rf "../stepcharts"
fi
mkdir "../stepcharts"

if [ -d "../audiofiles" ]; then
	rm -rf "../audiofiles"
fi
mkdir "../audiofiles"

for f in $(find -name "*.sm"); do
	echo "Copying $f ..."
	cp "$f" "../stepcharts"
done
#echo $(ls -l ../stepcharts | wc -l) "files copied"

for f in $(find -name "*.mp3" -o -name "*.wav"); do
	ffmpeg -i $f -o ${f%%.*}
done

for f in $(find -name "*.ogg" -o -name "*.mp3" -o -name "*.wav"); do
	echo "Copying $f ..."
	cp "$f" "../audiofiles"
done

IFS="$OIFS"
