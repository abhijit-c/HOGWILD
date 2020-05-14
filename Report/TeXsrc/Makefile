JobName=Hogwild
TeX=pdflatex
LatexFlags= -jobname=$(JobName) --shell-escape

default:
	$(TeX) $(LatexFlags) main.tex
	bibtex $(JobName)
	$(TeX) $(LatexFlags) main.tex
	$(TeX) $(LatexFlags) main.tex
	make clean
	cp $(JobName).pdf ../

clean:
	rm -f $(JobName).log $(JobName).aux $(JobName).out $(JobName).toc
	rm -f $(JobName).bbl $(JobName).blg 
	rm -rf _minted-*
