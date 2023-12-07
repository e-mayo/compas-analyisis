from matplotlib import pyplot as plt
import seaborn as sns

def plotlineplot_properties_hued_by(dataset, x, props, remove_legend=False, **kwargs):
    """

    Example:
    --------
    >>> plotlineplot_properties_hued_by(df.query('heteroatoms <= 13'),x = 'heterocycles', c='#212121')
    >>> plotlineplot_properties_hued_by(df.query('heteroatoms <= 13'),x = 'heteroatoms', c='#2064AA')
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))
    axs = axs.flatten()
    for i, prop in enumerate(props):
        sns.lineplot(data=dataset, x=x, y=prop, err_style="bars", ax=axs[i], **kwargs)
        # if remove_legend: axs[i].legend_.remove()
        remove_legend = True
        # reove y label
        axs[i].set_ylabel('')
    fig.suptitle(f"Grouped by {x}")
    plt.tight_layout()
    return fig, axs


def plotkde_properties_hued_by(dataset, props, hue, remove_legend=False, **kwargs):
    fig, axs = plt.subplots(2, 3, figsize=(15, 5))
    axs = axs.flatten()
    for i, prop in enumerate(props):
        sns.kdeplot(data=dataset, x=prop, 
                    fill=True,
                    hue=hue,
                    warn_singular=False,
                    ax=axs[i],
                    **kwargs
                    )
        if remove_legend: axs[i].legend_.remove()
        remove_legend = True
    fig.suptitle(f"Color by {hue}")
    plt.tight_layout()
    return fig, axs