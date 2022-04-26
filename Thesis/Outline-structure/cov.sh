pandoc --citeproc --number-sections thesis-structure.md \
-V margin-left=20mm --number-sections \
--filter pandoc-tablenos \
-V fontsize=10pt \
--pdf-engine=xelatex \
-V mainfont="Times New Roman" \
-o thesis-structure.pdf