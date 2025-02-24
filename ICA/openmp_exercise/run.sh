#!/bin/bash
#
# This is the shell script to run the example locally.

# load modules 

# Run test interactively
bash thread_test.sh &>> log

if [ $? -eq 0 ]
then
  echo "PASS"
else
  echo "FAIL" 
fi

# clean up files
rm log


