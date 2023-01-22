#!/bin/bash

# ./glucose -var-decay=0.95 -cla-decay=0.99 ../dataset/formulas/countbitsrotate016.processed.cnf

# for fold in ../result/generation/*
# do
#     echo ${fold}
#     for formula in $fold/*
#     do
#         # echo ${formula}
#         ./glucose -var-decay=0.75 -cla-decay=0.99 $formula >> output.log
#     done
# done

name=$1
vdecay=$2
cdecay=$3

for formula in ../result/generation/$name/*
do
    # echo ${formula}
    ./glucose -var-decay=$vdecay -cla-decay=$cdecay $formula >> output.log
done