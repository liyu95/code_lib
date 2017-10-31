#!/usr/bin/env bash 

while [[ -n $1 ]]; do
        # echo "\$1=$1"
        case $1 in
                -f | --filename )       shift
                                                FILENAME=$1
                                                ;;                      
        esac
        shift
done