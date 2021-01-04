#!/bin/bash

# CazzoVID-19
# Update the files' database

dir1='World'
dir2='Italy'

url1='https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
url2='https://github.com/pcm-dpc/COVID-19.git'
url3='https://github.com/CSSEGISandData/COVID-19.git'
url4='https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'

if [ -d "$dir1" ]
then
	echo "Found " $dir1
else
	mkdir $dir1
	cd $dir1 
fi

cd $dir1
rm *'_latest.csv'
rm *'_Report.csv'
wget $url1
wget $url4
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

if [ -d "$dir1" ]
then
	echo "Found " $dir1
else
	mkdir $dir1
	cd $dir1
	git init 
	git remote add origin $url1
fi

cd $dir1
git pull origin master
cd ..


