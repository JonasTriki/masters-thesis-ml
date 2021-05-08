import numpy as np
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt


def plot_word_vectors(
    transformed_word_embeddings: np.ndarray,
    words: list,
    title: str,
    x_label: str,
    y_label: str,
    word_colors: np.ndarray = None,
    ax: plt.axis = None,
    show_plot: bool = True,
    interactive: bool = False,
    continuous_word_colors: bool = False,
) -> None:
    """
    Plots word vectors in transformed 2D space.

    Parameters
    ----------
    transformed_word_embeddings : np.ndarray
        Word embeddings transformed into 2D space.
    words : list
        List of words to plot
    title : str,
        Title to use for the plot.
    x_label : str,
        Label to use for the x-axis.
    y_label : str
        Label to use for the y-axis
    word_colors : np.ndarray, optional
        Numpy array consisting of unique labels for each word (i.e. cluster labels),
        (defaults to None).
    ax : plt.axis
        Matplotlib axis (defaults to None)
    show_plot : bool
        Whether or not to call plt.show() (defaults to True)
    interactive : bool
        Whether or not to make the visualization interactive
        using Plotly (defaults to False).
    continuous_word_colors : bool
        Whether or not to make the word color continuous (defaults to False).
    """
    if interactive:

        # Plot interactive plot
        fig = px.scatter(
            x=transformed_word_embeddings[:, 0],
            y=transformed_word_embeddings[:, 1],
            title=title,
            labels={"x": x_label, "y": y_label},
            color=[
                cluster_label if continuous_word_colors else str(cluster_label)
                for cluster_label in word_colors
            ]
            if word_colors is not None
            else None,
            hover_data={"word": words},
        )
        fig.show()
    else:
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(
            transformed_word_embeddings[:, 0],
            transformed_word_embeddings[:, 1],
            c=word_colors,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if show_plot:
            plt.show()


def configure_plotting_for_thesis() -> None:
    """
    Configures plotting for thesis by using "ticks" Seaborn theme and Serif font.
    """
    sns.set_theme(style="ticks", palette="colorblind")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    print("Plots configured for thesis!")
