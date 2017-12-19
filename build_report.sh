
\#!/bin/bash

cd report_src
pdflatex main
bibtex main
pdflatex main
pdflatex main
mv main.pdf ../report.pdf
rm -rf *.lof *.log *.lot *.dvi *.toc *.bbl *.blg *.aux
cd ..
