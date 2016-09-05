import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib



def gen_highlight_text(smiles, molecule_prediction, atom_predictions):
    min_pred = np.amin(atom_predictions)
    max_pred = np.amax(atom_predictions)

    norm = matplotlib.colors.Normalize(vmin=min_pred,vmax=max_pred)
    cmap = cm.get_cmap('plasma')

    plt_colors = cm.ScalarMappable(norm=norm,cmap=cmap)

    line_text = ''

    short_molecule_pred = "%.2f" % float(molecule_prediction)
    line_text += '<h1>'+short_molecule_pred+' ----> '

    for i,letter in enumerate(list(smiles)):
        color_rgba = plt_colors.to_rgba(atom_predictions[i])
        letter_rgb = (color_rgba[0],color_rgba[1],color_rgba[2])
        letter_hex = matplotlib.colors.rgb2hex(letter_rgb)
        letter_rgba = str(color_rgba[0])+','+str(color_rgba[1])+','+str(color_rgba[2])+',0.5'

        #background-color: rgba(0,0,0,.5)
        line_text += '<span style="background-color: '+letter_hex+'">'+letter+'</span>'

    return line_text

#we can exploit html's native coloring for our purposes here
OUTPUT = open('rnn_molecules/solubility_predictions_rnn.html', 'w')
OUTPUT.write("<!DOCTYPE html>\n<html>\n<body>\n")

INPUT = open('../../data/example_output/solubility_predictions_rnn.csv', 'r')
for line in INPUT:
    line = line.rstrip()
    line_list = line.split(',')
    smiles = line_list[0]
    molecule_prediction = line_list[1]
    atom_predictions = line_list[2].split(':')

    #convert these to floats and ints, respectively
    atom_predictions = [float(x) for x in atom_predictions]

    line_text = gen_highlight_text(smiles, molecule_prediction, atom_predictions)

    OUTPUT.write(line_text)


INPUT.close()
OUTPUT.write('</body>\n</html>')
OUTPUT.close()
