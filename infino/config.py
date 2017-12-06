CELL_TYPES = ['B_CD5', 'B_Memory', 'B_Naive', 'CD4_Central_Memory',
       'CD4_Effector_Memory', 'CD4_Naive', 'CD4_Th1', 'CD4_Th17',
       'CD4_Th2', 'CD4_Treg', 'CD8_Central_Memory', 'CD8_Effector',
       'CD8_Naive', 'unknown_prop']


ROLLUPS = {
    'B': [c for c in CELL_TYPES if c.startswith('B_')],
    'CD4 T': [c for c in CELL_TYPES if c.startswith('CD4_')],
    'CD8 T': [c for c in CELL_TYPES if c.startswith('CD8_')],
    'Unknown': [c for c in CELL_TYPES if c.startswith('unknown_')]
}