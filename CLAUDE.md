When writing in the .tex files, remember these things:

- Use \begin{align} or \begin{align*} for bigger, centered equations.
- Text should be written up to the equations then use "%" on one line between the end of the text and \begin{align}.
- Use "\," after an equation if there is a "," or "." after, to get a small gap to the sign.
- For math in line with text use $...$
- Try to not use too much ":". Reformulate unless ":" is very natural.
- After all edits are done, compile main.tex
- If, when you look through, see any of the above is not followed, please fix it and report it to me.
- In results, when referring to figures or tables, try not to say "as shown if figure X, males on averagehave longer wings". Instead, say "males on average have longer wings (Figure X)"

Figure style (keep one common thread across all thesis figures):

- All figures share one look, defined once in scripts/thesis_style.py. Import from it (configure_thesis_style, style_axes, TRAIT_COLORS, SEMANTIC_COLORS, PALETTE) instead of redefining the style per figure. The figures notebook (notebooks/thesis_figures.ipynb) applies it once at the top.
- Font: serif (Times New Roman) with STIX math. Axes: seaborn whitegrid, no top/right spines, faint grid. Qualitative cycle: seaborn "colorblind".
- Colour by trait whenever the three traits appear in separate panels or series, using the fixed TRAIT_COLORS: body mass = blue #4C78A8, tarsus length = green #59A14F, wing length = orange #F28E2B.
- Use the accent pair SEMANTIC_COLORS["observed"] (blue #4C78A8) / ["adjusted"] (red #E45756) only when two series share one panel and must be told apart by colour (e.g. male vs female overlaid). Reference lines: dark grey.
- Export PDF at 600 dpi; full-width figures use width 6.7 in.