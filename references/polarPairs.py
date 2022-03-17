import logging

from pymol import cmd


def polar_pairs(sel1, sel2,  name, cutoff=4.0, angle=63.0):
    """
    Adapted from Thomas Holder see: https://sourceforge.net/p/pymol/mailman/message/27699323/

    Get the atoms number which distance is less or equal to a cut-off in Angstroms.
    SEE ALSO:
        cmd.find_pairs, cmd.distance

    :param sel1: the selection of atoms for the first chain
    :type sel1: str
    :param sel2: the selection of atoms for the second chain
    :type sel2: str
    :param name: the output object name.
    :type name: str
    :param cutoff: the cut-off in Angstroms.
    :type cutoff: float
    :param angle: h-bond angle cutoff in degrees. If angle="default", take "h_bond_max_angle" setting. If angle=0, do
    not detect h-bonding.
    :type angle: float
    :return: the list of lists of contacts: [[atom1_chainA_serial-number, atom2_chainA_serial-number, distance], ...]
    :rtype: list

    """
    cutoff = float(cutoff)
    if angle == 'default':
        angle = cmd.get('h_bond_max_angle', cmd.get_object_list(sel1)[0])
    angle = float(angle)
    logging.info("\tSettings: cutoff=%.1f angstrom, angle=%.1f degree" % (cutoff, angle))
    mode = 1 if angle > 0 else 0
    x = cmd.find_pairs('(%s) and donors' % sel1, '(%s) and acceptors' % sel2, cutoff=cutoff, mode=mode, angle=angle) + \
        cmd.find_pairs('(%s) and acceptors' % sel1, '(%s) and donors' % sel2, cutoff=cutoff, mode=mode, angle=angle)
    x = sorted(set(x))
    logging.info("\tFound %d polar contacts" % (len(x)))

    pairs_contacts = []
    for pair in x:
        distance = cmd.distance(name, '(%s`%s)' % pair[0], '(%s`%s)' % pair[1])
        # get the atoms numbers involved in the contact pairs and the distances between them: [atom_A, atom_B, dist]
        pairs_contacts.append([pair[0][1], pair[1][1], distance])
    return pairs_contacts


cmd.extend('polar_pairs', polar_pairs)

cmd.auto_arg[0].update({'polar_pairs': [cmd.selection_sc, 'selection', ', ']})
cmd.auto_arg[1].update({'polar_pairs': [cmd.selection_sc, 'selection', '']})
