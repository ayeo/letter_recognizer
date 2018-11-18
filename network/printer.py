import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class Printer(object):
    def print(self, Q, LR):
        cmap = ListedColormap(['w', 'r'])

        for x in range(np.size(Q, axis=0)):
          result = LR.guess(Q[x])
          plt.matshow(Q[x].reshape(6, 4), cmap=cmap)

          result = np.round(result, 4)
          data = [
            ['A', result[0]],
            ['B', result[1]],
            ['C', result[2]],
            ['D', result[3]],
            ['E', result[4]],
            ['F', result[5]],
          ]
          table = plt.table(cellText=data, cellLoc='center')
          plt.gca().axes.get_yaxis().set_visible(False)
          plt.gca().axes.get_xaxis().set_visible(False)
          table.set_fontsize(16)
          table.scale(1, 2)
          plt.savefig('images/' + str(x + 1) + '.png', bbox_inches='tight')
          #plt.show()