# Makefile for Agent Bazaar Paper

MAIN = root
TEX_FILES = $(wildcard sections/*.tex) root.tex
BIB_FILE = references.bib

# Default target: build the PDF and then clean up temporary files
all: $(MAIN).pdf
	$(MAKE) clean-soft

$(MAIN).pdf: $(TEX_FILES) $(BIB_FILE)
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

# Remove everything including the PDF
clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.pdf *.synctex.gz *.fdb_latexmk *.fls

# Remove only temporary compilation files, keep the PDF
clean-soft:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.synctex.gz *.fdb_latexmk *.fls

.PHONY: all clean clean-soft
