#/bin/bash
OIFS="$IFS"
IFS=$'\n'

cd $1

if [ -f "index.txt" ]; then
	rm index.txt
fi

for f in $(find -name "*.pkl"); do
	STR=${f:2}
	echo ${STR%.*} >> index.txt
done

IFS="$OIFS"
