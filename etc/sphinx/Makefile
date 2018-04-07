# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = cvopt
SOURCEDIR     = source
BUILDDIR      = ../../docs
MODULENAME    = cvopt
EXCLUDEMEMBERS= __init__

#export SPHINX_APIDOC_OPTIONS=members
export SPHINX_APIDOC_OPTIONS=

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

.PHONY: sphinx-apidoc
sphinx-apidoc:
	rm -rf source/modules/*
	sphinx-apidoc -e -f -o source/modules -M "../../$(MODULENAME)"
	rm source/modules/modules.rst
	rm source/modules/$(MODULENAME).rst
	for f in `find source/modules -name "*.rst"`; do\
		fname=`echo $$f | rev | cut -d "/" -f 1 | rev | sed "s/.rst//"| rev | cut -d "." -f 1 | rev`;\
		full_fname=`echo $$f | rev | cut -d "/" -f 1 | rev | sed "s/.rst//"`;\
		sed -i -e "s/package//" "$$f";\
		#sed -i -e "4s/^/Modules\n/" "$$f";\
		#sed -i -e "5s/^/----------\n/" "$$f";\
		#echo ".. toctree::" >> "$$f";\
		#sed -i -e "/.. toctree::/a\   :glob:" "$$f";\
		#sed -i -e "/   :glob:/a\   :maxdepth: 2" "$$f";\
		#sed -i -e "/   :maxdepth: 2/a\\\n   $$fname/*\\n" "$$f";\
		#echo ".. automodule::" >> "$$f";\
		#sed -i -e "/.. automodule::/a\    :inherited-members:" "$$f";\
		#sed -i -e "/.. automodule::/a\    :members:" "$$f";\
		#sed -i -e "/.. automodule::/a\    :exclude-members: $(EXCLUDEMEMBERS)" "$$f";\
		echo ".. autosummary::" >> "$$f";\
		sed -i -e "/.. autosummary::/a\   :template: mytemplate.rst" "$$f";\
		sed -i -e "/.. autosummary::/a\   :toctree: $$fname\\n" "$$f";\
		modnames=`./find_modules.py "module" "../../" "$$full_fname"`;\
		for modname in $$modnames; do\
			echo "   $$full_fname.$$modname" >> "$$f";\
		done;\
	done

.PHONY: html
# html:
html: sphinx-apidoc
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."
	# for github
	touch $(BUILDDIR)/.nojekyll

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)