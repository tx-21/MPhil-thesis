pandoc --citeproc \
--csl toolbox/csl/water-research.csl \
--bibliography toolbox/bib/ref.bib -M reference-section-title="Reference" \
-M link-citations=false Thesis/quick-notes/ref/ref-generator.md -o Thesis/quick-notes/ref/ref.tex