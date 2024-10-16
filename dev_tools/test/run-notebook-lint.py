#!/usr/bin/env python3
import glob
import subprocess


def main():
    out = 0
    for filename in glob.glob("docs/source/notebooks/*.ipynb"):
        ret = subprocess.run(["poetry", "run", "nbqa", "flake8", str(filename),
                              "--max-line-length", "120",
                              "--ignore", "E731",
                              "--select", "E,W,F",
                              "--nbqa-process-cells", "run"],
                             ).returncode
        if ret != 0:
            out = ret
    return out


if __name__ == "__main__":
    exit(main())

# OUT=0
# for filename in docs/source/notebooks/*.ipynb; do
#     poetry run nbqa flake8 $filename --max-line-length 120 --ignore E731 --select E,W,F --nbqa-process-cells run
#     RET=$?
#     if [ $RET -ne 0 ]
#     then
#         OUT=$RET
#     fi
# done
# exit $OUT
