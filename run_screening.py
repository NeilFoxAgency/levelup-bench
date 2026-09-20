import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import sys
sys.path.insert(0, '/Users/satoshinakamoto/Downloads/LevelUp Bench')

from src.levelup.experiments.milestone6_phase2_screening_driver import main
import sys

sys.argv = [
    'milestone6_phase2_screening_driver',
    '--manifest-path', 'experiments/milestone6_phase2_screening_readiness.json',
    '--manifest-sha256', '2ba74db3fd11bc0a2e8c707b9ad6e50ce1ce3aeb0eee04828487eaa8cde19eb1',
    '--raw-root', '/Users/satoshinakamoto/screening_runtime_v2',
    '--repository', '.',
    '--preparation-commit', 'f9641d0698fab4b68a2f52f72d6e4dad245b2d3b',
    '--validate-only'
]

main()
