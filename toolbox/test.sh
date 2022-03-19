pandoc --citeproc --number-sections \
--csl toolbox/csl/water-research.csl \
--bibliography toolbox/bib/ref.bib -M reference-section-title="Reference" \
-M link-citations=true toolbox/test.md -o toolbox/test.docx