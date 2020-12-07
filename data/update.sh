#!/bin/bash

dir1='CountryInfo'
dir2='Italy'
dir3='World'

file1='https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'

cd $dir1
rm *.csv
wget $file1
cd ..

cd $dir2
git pull origin master
cd ..

cd $dir3
git pull origin master
cd ..
