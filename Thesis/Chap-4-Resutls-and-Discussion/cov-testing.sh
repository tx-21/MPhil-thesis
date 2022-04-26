pandoc \
-o testing.pdf \
--pdf-engine=xelatex \
-V margin-left=20mm --number-sections \
--filter pandoc-tablenos \
--filter pandoc-fignos \
-V fontsize=10pt \
-V mainfont="Times New Roman" \
-M tablenos-number-by-section=False \
0422-pre.md \
