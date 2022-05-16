pandoc \
-o 0511.pdf \
--pdf-engine=xelatex \
-V margin-left=20mm --number-sections \
--filter pandoc-tablenos \
--filter pandoc-fignos \
-V fontsize=10pt \
-V mainfont="Times New Roman" \
-M tablenos-number-by-section=False \
0511.md \
