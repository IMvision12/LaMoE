# LaMoE architecture explorer

Open `index.html` in a local browser after downloading the source. The file is
self-contained and works offline. Select the model/hybrid, MLA or MoE view;
click a module or focus it and press Enter/Space for source-derived attributes.
Search highlights matching modules. Scroll or use buttons to zoom, drag to pan,
and select **Fit graph** to reset the view.

The main repository README embeds `model.svg`; `model.png` is a high-resolution
static export. `mla.svg` and `moe.svg` expand those modules. PNG equivalents are
included. SVG/PNG files have no network dependency or executable model code.

The visual style is inspired by [HF Viewer](https://hfviewer.com/). The graphs
were authored from LaMoE's source and configuration; they are not runtime traces,
HF Viewer exports, or measurements of latency, FLOPs or traced operation counts.
The expanded decoder view shows a hybrid block. Other blocks follow the layer
schedule shown in the overview. Both MLA and GDN run in a hybrid block; the FFN
has its own independent top-2 expert router.

The diagram-generation helper is maintained privately outside this repository.
PNG files are separate static exports. Model source remains the authority for
implementation behavior.
