#!/bin/bash

nums=(006 010 012)

for num in ${nums[*]}
do
	python structure_metric.py -o -nd '../../Documents/Research/Data_Repository/20150126_T4_135/20150126_T4_135_1H/20150126_T4_135_1H_'$num'.Tif'
done
