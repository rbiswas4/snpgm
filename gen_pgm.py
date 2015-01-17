#!/usr/bin/env python

from matplotlib import rc
from daft import PGM, Node, Plate
rc("font", family="serif", size=12)
rc("text", usetex=True)


pgm = PGM([6, 4.2], origin=[0., 0.2], observed_style='inner')

# x_1 and c distributions on top line
pgm.add_node(Node("sigdist", r"$\sigma_{\mathrm{int}}^{\mathrm{dist}}$",
                  3, 4))
pgm.add_node(Node("x1dist", r"$x_1^{\mathrm{dist}}$", 4, 4))
pgm.add_node(Node("cdist", r"$c^{\mathrm{dist}}$", 5, 4))

# Per-SN parameters:  top line in the plate
pgm.add_node(Node("x1itrue", r"$x_{1,i}^\mathrm{true}$", 4, 3))
pgm.add_node(Node("citrue", r"$c_i^\mathrm{true}$", 5, 3))

# Per-SN parameters: second line in the plate
pgm.add_node(Node("x0itrue", r"$x_{0,i}^\mathrm{true}$", 3, 2))
#pgm.add_node(Node("mui", r"$\mu_i$", 2, 2))

# Per-SN parameters: third line in the plate
pgm.add_node(Node("zi", r"$z_i$", 2, 1, observed=True))

# Observed photometry
pgm.add_node(Node("fij", r"$f_{i,j}$", 4, 1, observed=True))

pgm.add_node(Node("t0true", r"$t_0^{\mathrm{true}}$", 5, 1))

# Big Plate: SNe
pgm.add_plate(Plate([1.5, 0.5, 4, 3.],
                    label=r"SNe $i = 1, \cdots, N_{SN}$",
                    shift=-0.1))

# Cosmological parameters
pgm.add_node(Node("cosmology", r"$\Omega$", 1, 2))

# nuisance parameters
pgm.add_node(Node("nuisance",
                  r"$\alpha, \beta, x_{00}$",
                  0.7, 3, aspect=2.0))

# Add in the edges.
pgm.add_edge("x1dist", "x1itrue")
pgm.add_edge("cdist", "citrue")
pgm.add_edge("sigdist", "x0itrue")

pgm.add_edge("x1itrue", "x0itrue")
pgm.add_edge("citrue", "x0itrue")

pgm.add_edge("cosmology", "x0itrue")
pgm.add_edge("nuisance", "x0itrue")

pgm.add_edge("zi", "x0itrue")

pgm.add_edge("x0itrue", "fij")
pgm.add_edge("x1itrue", "fij")
pgm.add_edge("citrue", "fij")
pgm.add_edge("zi", "fij")
pgm.add_edge("t0true", "fij")

# Render and save.
pgm.render()
pgm.figure.savefig("snpgm.png", dpi=150)
