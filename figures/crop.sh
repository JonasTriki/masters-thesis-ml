#!/bin/bash

# Delete old files
rm -rf *_cropped.pdf

# Crop all files
for pdf_file in *.pdf; do
    pdf-crop-margins -s -u $pdf_file
done