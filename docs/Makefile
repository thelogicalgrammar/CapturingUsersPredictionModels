SPHINXBUILD = sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build
DOCTREESDIR = $(BUILDDIR)/doctrees

html:
	$(SPHINXBUILD) -b html -d $(DOCTREESDIR) $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)

.PHONY: html clean
