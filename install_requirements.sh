#!/bin/bash
pip install uv
uv pip install --system -r requirements_base.txt
python check_gpu.py