cd /Users/joaorietra/Developer/hidrometros/local/exp
R() { n=$1; shift; PYTHONPATH=../../src ../../.venv/bin/python exp.py "$@" > logs/$n.log 2>&1; }
R e0_protocol --epochs 10
R e3_flat --arch flat
R e4_pool3flat --arch pool3flat
R e5_gap4x3 --arch pool3gap2x2
R e6_gapmax3 --arch gap_maxpool3
R e7_gap_lr1e2 --lr 1e-2
R e8_gap_nonorm --norm none
