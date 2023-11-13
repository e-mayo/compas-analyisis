from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D
from IPython.display import SVG
from typing import Union, List, Tuple, Dict, Any

def validate_label(label):
    ...

def create_drawing_canvas(width, height):
    ...

def draw_molecule(d2d, mol, label, label_offset, **kwargs):
    ...

def draw_labels_on_mol(d2d, labels, ring_center, offset):
    ...
    return SVG(d2d.GetDrawingText())

def draw_traverse_direction_on_mol(mol, traversal, ring_center, colors):
    ...
    return SVG(d2d.GetDrawingText())

def show_mol(mol:Chem.Mol, 
            label:Union[str,None] = None,
            label_offset:float = 0.0,
            notes:List = None,
            notes_offset:float = 0.4,
            **kwargs) -> str:
    """
    Show molecule with labels on atoms, bonds or rings.

    Parameters:
    -----------
    mol: Chem.Mol
        The RDKit mol object to draw.
    label: str
        Entity to label ('atoms', 'bonds' or 'rings')
    kwargs: dict
        argumetns to be passed to the RDKit drawing function

    Return:
    -------
    SVG: str
        SVG drawing text
    """
    # validate input
    if label not in ['atoms', 'bonds', 'rings', None]:
        raise ValueError(f"Invalid label '{label}'.")

    # prepare canva for drawing
    d2d = rdMolDraw2D.MolDraw2DSVG(500, 300)
    # get drawing options
    dopts = d2d.drawOptions()
    # select case 'atoms', 'bonds' or 'ring'
    if not label:
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, **kwargs)
    elif label == 'atoms':
        dopts.addAtomIndices = True
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, **kwargs)
    elif label == 'bonds':
        dopts.addBondIndices = True
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, **kwargs)
    elif label == 'rings':
        centers = get_ring_centers(mol) 
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, **kwargs)
        for i, coord in enumerate(centers):
            d2d.DrawString(f"{i}", Point2D(*coord+label_offset))
            d2d.DrawString(f"+", Point2D(0,0))
    if notes:
        centers = get_ring_centers(mol) 
        for i, coord in enumerate(centers):
            d2d.DrawString(f"{notes[i]}", Point2D(*coord+notes_offset))    
    d2d.FinishDrawing()
    return SVG(d2d.GetDrawingText())

def draw_traverse_direction_on_mol(mol, traversal, ring_center):
    colors = [
    # dark black
    (0.23529412, 0.23529412, 0.23529412, 1),
    # dark green)
    (0.21960784, 0.4627451 , 0.11372549, 1),
    # dark blu)
    (0.06666667, 0.5372549 , 0.63529412, 1),
    # weird orang)
    (0.90196078, 0.56862745, 0.21960784, 1),
    # dark pink
    (0.90196078, 0.21960784, 0.21960784, 1),
    # dark black
    (0.23529412, 0.23529412, 0.23529412, 1),
    # dark green)
    (0.21960784, 0.4627451 , 0.11372549, 1),
    # dark blu)
    (0.06666667, 0.5372549 , 0.63529412, 1),
    # weird orang)
    (0.90196078, 0.56862745, 0.21960784, 1),
    # dark pink
    (0.90196078, 0.21960784, 0.21960784, 1)
    ]
    
    d2d = rdMolDraw2D.MolDraw2DSVG(800, 600)
    # draw molecule 
    rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol)
    path_id = 0
    
    for node in traversal.nodes():
        if not list(traversal.successors(node)):continue
        # get the node center
        center = list(ring_center[node])
        # get the successor center
        path_id = 2
        for succ in list(traversal.successors(node))[::-1]:
            # draw an arrow from node to succ
            succ_center = ring_center[succ]
            d2d.DrawArrow(Point2D(*center), Point2D(*succ_center),frac=0.1,asPolygon=True,color=colors[path_id])
            path_id = 0
    d2d.FinishDrawing()
    return SVG(d2d.GetDrawingText())