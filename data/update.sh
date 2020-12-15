#!/bin/bash

"
 CazzoVID-19
 Update the files' database
"

dir1='CountryInfo'
dir2='Italy'
dir3='World'

file1='https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
url2='https://github.com/pcm-dpc/COVID-19.git'
url3='https://github.com/CSSEGISandData/COVID-19.git'

if [ -d "$dir1" ]
then
	echo "Found " $dir1
else
	mkdir $dir1
	cd $dir1 
fi

cd $dir1
rm *.csv
wget $file1
cd ..

if [ -d "$dir2" ]
then
	echo "Found " $dir2
else
	mkdir $dir2
	cd $dir2
	git init 
	git remote add origin $url2
fi

cd $dir2
git pull origin master
cd ..

if [ -d "$dir3" ]
then
	echo "Found " $dir3
else
	mkdir $dir3
	cd $dir3
	git init 
	git remote add origin $url3 
fi

cd $dir3
git pull origin master
cd ..


