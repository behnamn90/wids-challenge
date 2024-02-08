import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_missing_summary(df):
    total = df.shape[0]
    missing_percent = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        per = (null_count/total) * 100
        missing_percent[col] = per
    return dict(sorted(missing_percent.items(), key=lambda item: item[1], reverse=True))

def plot_distributions_by_target(df, binary_col):
    # Filter out the binary column from the plotting list
    plot_columns = df.columns.drop(binary_col)
    color_palette = 'seismic'

    # Determine the number of rows needed for subplots, given 4 columns
    n = len(plot_columns)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, nrows*5))
    axs = axs.flatten()  # Flatten to 1D array for easy iteration
    
    # Get unique classes from binary column
    classes = df[binary_col].dropna().unique()
    
    for i, column in enumerate(plot_columns):
        ax = axs[i]
        
        # Text box initialization
        textstr = ''
        for cls in classes:
            class_mask = df[binary_col] == cls
            nan_percent = df.loc[class_mask, column].isna().mean() * 100
            textstr += f'{cls} Missing: {nan_percent:.2f}%\n'
        
        if pd.api.types.is_numeric_dtype(df[column]):
            # Plot numerical data
            sns.histplot(df, x=column, hue=binary_col, bins='auto', kde=False, ax=ax, palette=color_palette, alpha=0.5, element="step", stat="density", common_norm=False)
        else:
            # Normalize and plot categorical data
            category_proportions = df.groupby(binary_col)[column].value_counts(normalize=True).rename("proportion").reset_index()
            
            # Truncate category names longer than 10 characters for display
            category_proportions[column] = category_proportions[column].astype(str).apply(lambda x: x[:10] + "..." if len(x) > 10 else x)
            sns.barplot(x=column, y="proportion", hue=binary_col, data=category_proportions, ax=ax, palette=color_palette)
            ax.tick_params(axis='x', rotation=45)
        
        ax.set_title(f'{column}')
        ax.set_ylabel('Density' if pd.api.types.is_numeric_dtype(df[column]) else 'Proportion')
        ax.set_xlabel(column)#column[:10] + "..." if len(column) > 10 else column)
        
        # Add textbox for percentage of missing values per class
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr.rstrip(), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Only show legend on the first subplot
        if i != 0:
            ax.legend().set_visible(False)
    
    # Hide any unused subplots
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()