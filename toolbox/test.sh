pandoc --citeproc --number-sections \
--csl csl/water-research.csl \
--bibliography bib/ref.bib -M reference-section-title="Reference" \
-M link-citations=true test.md -o test.pdf