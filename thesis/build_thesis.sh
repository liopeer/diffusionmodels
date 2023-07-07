#!/bin/bash

pdflatex main.tex
biber main
pdflatex main.tex

mv main.pdf /app/write_out/