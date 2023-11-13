import matplotlib
colors = [
        [0, "#0082C1"], # 
        [0.5, "#A768A8"], # 
        [1, "#730B6E"], # 
            ]
# colors = [
#         [0, "#570000"], # 
#         [1, "#00D5FF"], # 
#             ]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


heterocycles_colors = {
    'benzene':'#686868',            # aromatic
    'cyclobutadiene':'#686868',     # antiaromatic
    'borole':'#00D8C9',             # antiaromatic
    'borinine':'#00B5A9',           # aromatic
    '14diborinine':'#00948B',       # aromatic 
    'dhdiborinine':'#00746D',       # antiaromatic
    'pyrrole':'#0099FF',            # aromatic
    'pyridine':'#077AD5',           # aromatic
    'pyrazine':'#015CAC',           # aromatic
    'furan':'#CF252E',              # aromatic
    'thiophene':'#FBD55C',          # aromatic

}