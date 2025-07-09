"""
Funciones para graficar métricas de evaluación.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Grafica la matriz de confusión.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(cm)),
        yticks=range(len(cm)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='Etiqueta real',
        xlabel='Etiqueta predicha'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    plt.title('Matriz de confusión')
    plt.tight_layout()
    plt.show()