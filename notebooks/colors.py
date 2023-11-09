import matplotlib
colors = [
        [0, "#0082C1"], # 
        [0.5, "#A768A8"], # 
        [1, "#730B6E"], # 
            ]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
colors = [
        [0, "#570000"], # 
        [1, "#00D5FF"], # 
            ]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


heterocycles_colors = {
    'benzene':'#686868',            # aromatic
    'cyclobutadiene':'#686868',     # antiaromatic
    'pyrrole':'#0099FF',            # aromatic
    'borole':'#00D8C9',             # antiaromatic
    'furan':'#CF252E',              # aromatic
    'thiophene':'#FBD55C',          # aromatic
    'dhdiborinine':'#00746D',       # antiaromatic
    '14diborinine':'#00948B',       # aromatic 
    'pyrazine':'#015CAC',           # aromatic
    'pyridine':'#077AD5',           # aromatic
    'borinine':'#00B5A9',           # aromatic

}