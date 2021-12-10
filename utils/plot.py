import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def show_rul_distribution(df, ax, subset=1, split='train', unit_col='unit', rul_col='RUL'):
    """
    Show distribution of remaining useful life for train and test sets.
    For the training set, this shows the maximum RUL value, which is
    equivalent to the entire unit life.
    For the test set, this shows the target RUL value.
    """
    # Group dataframe by unit
    df_rul = df[[unit_col, rul_col]].groupby(unit_col)
    # For test set, only take last time sample
    if (split.lower() == 'test'):
        df_rul = df_rul.last().reset_index()
        title = 'Final RUL Distribution for {} Set - FD00{}'
    # For training set, take max RUL / RUL at first time sample
    else:
        df_rul = df_rul.max(rul_col).reset_index()
        title = 'Max-RUL Distribution for {} Set - FD00{}'
    # Plot histogram
    df_rul[rul_col].hist(bins=15, ax=ax)
    ax.set_xlabel('RUL')
    ax.set_ylabel('frequency')
    ax.set_title(title.format(split.title(), subset))

    return ax


def show_sensor(df, sensor, ax, unit_col='unit'):
    """Plot evolution of single sensors for all units."""
    # Plot sensor for each unit
    for i in df[unit_col].unique():
        ax.plot('RUL', sensor, data=df[df[unit_col] == i])
    ax.set_ylabel(sensor)
    return ax


def show_sensors(df, sensors, unit_col='unit'):
    """Plot evolution of all sensors for all units."""
    # Setup figure
    fig, axes = plt.subplots(7, 3, sharex=True, figsize=(20, 10))
    axes = axes.ravel()
    # Plot each sensor
    for ax, sensor in zip(axes, sensors):
        ax = show_sensor(df, sensor, ax, unit_col)
    # Set axes limits and labels
    axes[-2].set_xlim(250, 0)
    axes[-2].set_xticks(np.arange(0, 275, 25))
    axes[-2].set_xlabel('Remaining Useful Life')
    axes[1].set_title('Sensor Evolution with RUL', fontsize=15)
    
    return fig, axes


def show_correlation(df, unit=1.0, n_meta=5):
    # Take sensor columns and compute correlation
    corr = df[df.unit == 1.0].iloc[:, n_meta:].corr()
    # Set correlation style as heatmap
    corr = corr.style.background_gradient(cmap='coolwarm')
    return corr
