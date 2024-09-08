import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(data, column_name, title_name):
    plt.figure(figsize=(10, 6), dpi=500)
    ax = sns.countplot(x=column_name, data=data)

    plt.xlabel('Weld Process', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(title_name, fontsize=16)
    total = len(data[column_name])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='top', fontsize=10, color='black', xytext=(0, 10),
                    textcoords='offset points')

    for p in ax.patches:
        count_value = '{:.1f}'.format(p.get_height())
        ax.annotate(count_value, (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 10),
                    textcoords='offset points')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_finalscore_distribution(df, column_name, title):
    data = df[column_name]

    # Filtering out non-numeric values
    numeric_data = pd.to_numeric(data, errors='coerce').dropna()

    # Plotting the histogram
    plt.figure(figsize=(13, 6), dpi=600)
    plt.subplot(1, 2, 1)
    plt.hist(numeric_data, color='darkcyan', bins=30)
    plt.title(f'Distribution of {column_name} variable')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')

    # Adding vertical lines for mean, median, and mode
    plt.axvline(numeric_data.mean(), color='red', linestyle='solid', linewidth=2, label=f'Mean: {numeric_data.mean():.2f}')
    plt.axvline(numeric_data.median(), color='red', linestyle='dashed', linewidth=2, label=f'Median: {numeric_data.median():.2f}')
    plt.axvline(numeric_data.mode().iloc[0], color='red', linestyle='dotted', linewidth=2, label=f'Mode: {numeric_data.mode().iloc[0]:.2f}')
    plt.legend()

    # Plotting the Kernel Density Plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data, fill=True, color='darkcyan')

    # Adding vertical lines for mean, median, and mode
    plt.axvline(data.mean(), color='red', linestyle='solid', linewidth=2, label=f'Mean: {data.mean():.2f}')
    plt.axvline(data.median(), color='red', linestyle='dashed', linewidth=2, label=f'Median: {data.median():.2f}')
    plt.axvline(data.mode().iloc[0], color='red', linestyle='dotted', linewidth=2, label=f'Mode: {data.mode().iloc[0]:.2f}')

    # Adding title and labels
    plt.title(f'Kernel Density Plot of {column_name} with Mean, Median, and Mode')
    plt.xlabel(column_name)
    plt.ylabel('Density')

    # Adding legend
    plt.legend()

    # Displaying the plot
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()
def plot_pass_fail_histogram(df, col_name, title):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), dpi=600)

    # Plotting for failed results
    df[df['result'] == 'fail'][col_name].plot.hist(ax=ax[0], bins=15, edgecolor='black', color='coral')
    x1 = list(range(1, 16, 1))
    ax[0].set_xticks(x1)
    ax[0].set_title(f'{title} - Failed')

    # Plotting for passed results
    df[df['result'] == 'pass'][col_name].plot.hist(ax=ax[1], color='skyblue', bins=15, edgecolor='black')
    x2 = list(range(1, 16, 1))
    ax[1].set_xticks(x2)
    ax[1].set_title(f'{title} - Passed')

    plt.show()
