# ito-research/moonsplat:poc = the research image (research/systems/moonsplat/Dockerfile.ito:
# MoonSplat 86c7e15 on the mast3r-slam image, MASt3R-SLAM backend as a thread) plus the Python
# packages its main.py and Scaffold-GS model import that the image lacks.
#   docker build -t ito-research/moonsplat:poc tools/poc/moonsplat
FROM ito-research/moonsplat:latest
RUN pip install --no-cache-dir munch==4.0.0 jaxtyping==0.2.36 \
 && pip install --no-cache-dir torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cu124.html \
 && python -c "import torch_scatter, munch, jaxtyping"
