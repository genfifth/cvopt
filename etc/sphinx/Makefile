# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = cvopt
SOURCEDIR     = source
BUILDDIR      = ../../docs
MODULENAME    = cvopt
#EXCLUDEMEMBERS= 

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

.PHONY: sphinx-apidoc
sphinx-apidoc:
	sphinx-apidoc -f -o source/modules -M "../../$(MODULENAME)"
	rm source/modules/modules.rst
	rm source/modules/$(MODULENAME).rst
	for f in `find source/modules -name "*.rst"`; do\
 		sed -i -e "s/package//" "$$f";\
		#sed -i -e "/.. automodule::/a\    :inherited-members:" "$$f";\
		#sed -i -e "/.. automodule::/a\    :exclude-members: $(EXCLUDEMEMBERS)" "$$f";\
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