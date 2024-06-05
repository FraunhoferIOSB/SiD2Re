#!/usr/bin/env bash
OUT=0
for filename in docs/source/notebooks/*.ipynb; do
    poetry run nbqa flake8 $filename --max-line-length 120 --ignore E731 --select E,W,F --nbqa-process-cells run
    RET=$?
    if [ $RET -ne 0 ]
    then
        OUT=$RET
    fi
done
exit $OUT
