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
    'benzene':'#E1E1E1',            # aromatic
    'cyclobutadiene':'#9E9E9E',     # antiaromatic
    'borinine':'#00665D',           # aromatic
    '14diborinine':'#34988F',       # aromatic 
    'dhdiborinine':'#80CDC2',       # antiaromatic
    'borole':'#C7EBE6',             # antiaromatic
    'pyridine':'#2066AC',           # aromatic
    'pyrazine':'#4294C3',           # aromatic
    'pyrrole':'#93C6DF',            # aromatic
    'furan':'#E44591',              # aromatic
    'thiophene':'#F9D55C',          # aromatic

}

kde_kwargs = {'palette':cmap,
            'alpha':0.4,
            'edgecolor':'#212121',
            'common_norm':False,
}