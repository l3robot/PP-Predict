import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

def display2DGridSearch(scores, hyp1, hyp2, hyp1_name=None, hyp2_name=None): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(hyp1, hyp2, scores, cmap=cm.coolwarm)
    if hyp1_name != None and hyp2_name != None:
        ax.set_xlabel(hyp1_name)
        ax.set_ylabel(hyp2_name)
    ax.set_zlabel('Scores en validation')
    plt.show()

def save2DGridSearch(scores, hyp1, hyp2, hyp1_name=None, hyp2_name=None, file='results'): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(hyp1, hyp2, scores, cmap=cm.coolwarm)
    if hyp1_name != None and hyp2_name != None:
        ax.set_xlabel(hyp1_name)
        ax.set_ylabel(hyp2_name)
    ax.set_zlabel('Scores en validation')
    fig.savefig('{}.png'.format(file))

def displayGridSearch(scores, hyp1, hyp1_name=None): 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(hyp1, scores)
    if hyp1_name != None:
        ax.set_xlabel(hyp1_name)
    ax.set_ylabel('Scores en validation')
    plt.show()