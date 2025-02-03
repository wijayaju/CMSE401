#!/bin/bash

for USERNAME in $(who | cut -d " " -f 1);
do
	echo $(squeue -l -u ${USERNAME});
done
