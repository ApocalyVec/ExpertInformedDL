#!/bin/bash

# unpack files from Google Drive download, for folders that are too large and have been split into multiple zip files

# enter the folder containing the zip files
cd C:/Users/apoca/Downloads/temp

# loop through all the zip files
for zip_file in *.zip
do
    # extract the contents of the zip file
    unzip "$zip_file"
done