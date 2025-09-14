
from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
import numpy as np
import json
import squarify
from statsmodels.graphics.mosaicplot import mosaic


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        analysis_type = request.form['analysisType']
        columns = request.form.getlist('columns')
        return redirect(url_for('generate_plot', filename=file.filename, analysis_type=analysis_type, columns=','.join(columns)))
    return redirect(url_for('index'))


@app.route('/plot')
def plot():
    filename = request.args.get('filename')
    analysis_type = request.args.get('analysis_type')
    columns = request.args.get('columns').split(',')

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)



# Function to determine plot types
def plot_types(col1, col2=None):
    if col2 is None:
        if pd.api.types.is_numeric_dtype(col1):
            return 'quantitative'
        elif pd.api.types.is_categorical_dtype(col1) or pd.api.types.is_object_dtype(col1):
            return 'categorical'
    else:
        if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
            return 'quant vs. quant'
        elif not pd.api.types.is_numeric_dtype(col1) and not pd.api.types.is_numeric_dtype(col2):
            return 'categ vs. categ'
        elif not pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
            return 'categ vs. quant'
        elif pd.api.types.is_numeric_dtype(col1) and not pd.api.types.is_numeric_dtype(col2):
            return 'quant vs. categ'
    return 'unknown'



@app.route('/generate_plot')
def generate_plot():
    filename = request.args.get('filename')
    analysis_type = request.args.get('analysis_type')
    columns = request.args.get('columns').split(',')

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)
    plot_urls = []

    if analysis_type.lower() == 'univariate' or len(columns) == 1:
      
        for col1 in columns:
            plot_type = plot_types(df[col1])
            
            if plot_type == 'quantitative':
                plot_urls.append(histogram_plot(df, col1))            
                plot_urls.append(ecd_plot(df, col1))
                plot_urls.append(area_chart(df, col1))
                plot_urls.append(lollipop_chart(df, col1))
                plot_urls.append(dot_chart(df, col1))
            
            elif plot_type == 'categorical':
                plot_urls.append(count_plot(df, col1))            
                plot_urls.append(bar_chart(df, col1))
                plot_urls.append(bubble_chart(df, col1))
                plot_urls.append(pie_chart(df, col1))
                plot_urls.append(donut_chart(df, col1))


    elif analysis_type.lower() == 'bivariate' and len(columns) == 2:
        col1, col2 = columns
        plot_type = plot_types(df[col1], df[col2])

        if plot_type == 'quant vs. quant':
        
            plot_urls.append(scatter_plot(df, col1, col2))            
            plot_urls.append(line_plot(df, col1, col2))

        if plot_type == 'categ vs. categ':
        
            plot_urls.append(stacked_bar(df, col1, col2))            
            plot_urls.append(grouped_bar(df, col1, col2))
            plot_urls.append(segmented_bar(df, col1, col2))


        if plot_type == 'categ vs. quant':
            
            plot_urls.append(density_plot(df, col1, col2))            
            plot_urls.append(strip_plot(df, col1, col2))
            plot_urls.append(box_plot(df, col1, col2))
            plot_urls.append(boxen_plot(df, col1, col2))
            plot_urls.append(violin_plot(df, col1, col2))
            plot_urls.append(swarm_plot(df, col1, col2))


        if plot_type == 'quant vs. categ':
            
            plot_urls.append(density_plot(df, col1, col2))            
            plot_urls.append(strip_plot(df, col1, col2))
            plot_urls.append(box_plot(df, col1, col2))
            plot_urls.append(boxen_plot(df, col1, col2))
            plot_urls.append(violin_plot(df, col1, col2))
            plot_urls.append(swarm_plot(df, col1, col2))


    elif analysis_type.lower() == 'multivariate' or len(columns) == 2:

        col1, col2 = columns
        plot_type = plot_types(df[col1], df[col2])

        if plot_type == 'quant vs. quant':
        
            columns = [col for col in columns if col in df.columns]
        
            plot_urls.append(heatmap(df, columns))


        if plot_type == 'categ vs. categ':
            
            plot_urls.append(treemap(df, col1, col2))            
            plot_urls.append(mosaic_plot(df, col1, col2))       


        if plot_type in ['categ_vs_quant', 'quant_vs_categ']:
        
            plot_urls.append(density_plot(df, col1, col2))
            plot_urls.append(strip_plot(df, col1, col2))
            plot_urls.append(box_plot(df, col1, col2))
            plot_urls.append(boxen_plot(df, col1, col2))
            plot_urls.append(violin_plot(df, col1, col2))
            plot_urls.append(swarm_plot(df, col1, col2))

    return render_template('Plot.html', plot_urls=plot_urls, analysis_type=analysis_type, columns=columns)









# =================================================================================================== #
#                                   1. Quantitative vs. Quantitative                                  #
# =================================================================================================== #


# ------------------------------------------------------------------------------------ #
#                                Univariate Analysis                                   #
# ------------------------------------------------------------------------------------ #


# .................................................... #
#                  Histogram plot                      #
# .................................................... #

def histogram_plot(data, col1):
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col1], color='Rebeccapurple', edgecolor='Red')
    plt.title(f"Histogram plot of {col1}", color="red", size=17)
    plt.xlabel(f'{col1}', color="blue", size=12)
    plt.ylabel("Frequency", color="blue", size=12)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url

# .................................................... #
#                     ECD Line Plot                    #
# .................................................... #

def ecd_plot(data, col1):
    sorted_df = data.sort_values(by=col1)
    sorted_df['Cumulative count'] = range(1, len(sorted_df) + 1)
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=sorted_df[col1], y=sorted_df['Cumulative count'], marker='o', linestyle='-', color='Red', markersize=5)
    plt.title(f"Empirical Cumulative Distribution of {col1}", color="red", size=17)
    plt.xlabel(col1, color="blue", size=12)
    plt.ylabel("Count", color="blue", size=12)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url

# .................................................... #
#                       Area chart                     #
# .................................................... #

def area_chart(data, col1):
    class_count = data[col1].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    class_count.plot(kind='area', color="skyblue", alpha=1)
    plt.title(f"Area chart of {col1}", color="red", size=17)
    plt.xlabel(f'{col1}', color="blue", size=12)
    plt.ylabel("Count", color="blue", size=12)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url

# .................................................... #
#                   Lollipop chart                     #
# .................................................... #

def lollipop_chart(data, col1):
    col1_count = data[col1].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.stem(col1_count.index, col1_count.values)
    plt.title(f"Lollipop chart of {col1}", color="red", size=17)
    plt.xlabel(col1, color="blue", size=12)
    plt.ylabel("Count", color="blue", size=12)
    plt.grid(True)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url

# .................................................... #
#                       Dot chart                      #
# .................................................... #

def dot_chart(data, col1):
    data = data.dropna(subset=[col1])
    col1_counts = data[col1].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    for i, j in col1_counts.items():
        x_values = np.full(j, i)
        y_values = np.arange(1, j + 1)
        plt.scatter(x_values, y_values, color='yellow', edgecolor='black', s=100, alpha=0.7)
    plt.title(f'Dot chart of {col1}', color="red", size=17)
    plt.xlabel(col1, color="blue", size=12)
    plt.ylabel('Count', color="blue", size=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url




# ------------------------------------------------------------------------------------ #
#                                Bivariate Analysis                                    #
# ------------------------------------------------------------------------------------ #


# .................................................... #
#                    Scatter plot                      #
# .................................................... #

def scatter_plot(data, col1, col2):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=data[col1], y=data[col2], color='Red', edgecolor='Orange')
    plt.title(f'Scatter plot of {col1} vs. {col2}', color='red', size=17)
    plt.xlabel(f'{col1}', color='blue', size=12)
    plt.ylabel(f'{col2}', color='blue', size=12)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url



# .................................................... #
#                       Line plot                      #
# .................................................... #

def line_plot(data, col1, col2):
    plt.figure(figsize=(6, 4))
    plt.plot(data[col1], label=col1, color='Red')
    plt.plot(data[col2], label=col2, color='blue')
    plt.title(f'Line plot of {col1} vs. {col2}', color='red', size=17)
    plt.legend()
    plt.xlabel(f'{col1}', color='blue', size=12)
    plt.ylabel(f'{col2}', color='blue', size=12)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url


# .................................................... #
#                       HeatMap                        #
# .................................................... #

def heatmap(data, selected_columns):
    numeric_df = data[selected_columns].select_dtypes(include='number')
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, ax=ax)
    ax.set_title("Correlation Matrix (Correlogram)")

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url



# =================================================================================================== #
#                                   2. Categorical vs. Categorical                                    #
# =================================================================================================== #


# ------------------------------------------------------------------------------------ #
#                                Univariate Analysis                                   #
# ------------------------------------------------------------------------------------ #


# .................................................... #
#                      Pie chart                       #
# .................................................... #

def pie_chart(data, col1):
    count = data[col1].value_counts()
    
    plt.figure(figsize=(6, 4))
    plt.pie(count, labels=count.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title(f"Pie chart of {col1}")
    
    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url

# .................................................... #
#                    Donut chart                       #
# .................................................... #

def donut_chart(data, col1):
    count = data[col1].value_counts()
    
    plt.figure(figsize=(6, 4))
    plt.pie(count, labels=count.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'), wedgeprops=dict(width=0.5))
    plt.title(f"Donut chart of {col1}")
    
    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url



# ------------------------------------------------------------------------------------ #
#                                Bivariate Analysis                                    #
# ------------------------------------------------------------------------------------ #

# .................................................... #
#                 Stacked bar chart                    #
# .................................................... #

def stacked_bar(data, col1, col2):
    unique_values = data[col2].unique()
    positive_label = unique_values[1]  
    negative_label = unique_values[0]  

    positive = data[data[col2] == positive_label].groupby(data[col1]).size()
    negative = data[data[col2] == negative_label].groupby(data[col1]).size()

    count_df = pd.DataFrame({
        positive_label: positive,
        negative_label: negative
    }).reset_index()

    count_df.fillna(0, inplace=True)

    bar_width = 0.30
    index = np.arange(len(count_df))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(index, count_df[positive_label], bar_width, label=str(positive_label), color='cyan')
    ax.bar(index, count_df[negative_label], bar_width, bottom=count_df[positive_label], 
           label=str(negative_label), color='firebrick')

    ax.set_title("Stacked Bar Chart", color="red", size=17)
    ax.set_xlabel(col1, color="blue", size=12)
    ax.set_ylabel(col2, color="blue", size=12)

    ax.set_xticks(index)
    ax.set_xticklabels(count_df[col1])
    ax.legend(unique_values)

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url


# .................................................... #
#                 Grouped bar chart                    #
# .................................................... #


def grouped_bar(data, col1, col2):
    unique_values = data[col2].unique()
    positive_label = unique_values[1]  
    negative_label = unique_values[0]  

    positive = data[data[col2] == positive_label].groupby(data[col1]).size()
    negative = data[data[col2] == negative_label].groupby(data[col1]).size()

    count_df = pd.DataFrame({
        positive_label: positive,
        negative_label: negative
    }).reset_index()

    count_df.fillna(0, inplace=True)

    bar_width = 0.30
    index = np.arange(len(count_df))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(index - bar_width/2, count_df[positive_label], bar_width, label=str(positive_label), color='cyan')
    ax.bar(index + bar_width/2, count_df[negative_label], bar_width, label=str(negative_label), color='firebrick')

    ax.set_title("Grouped Bar Chart", color="red", size=17)
    ax.set_xlabel(col1, color="blue", size=12)
    ax.set_ylabel(col2, color="blue", size=12)

    ax.set_xticks(index)
    ax.set_xticklabels(count_df[col1])
    ax.legend(unique_values)

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url


# .................................................... #
#               Segmented bar chart                    #
# .................................................... #


def segmented_bar(data, col1, col2):
    group_count = data.groupby([col1, col2]).size().unstack().fillna(0)
    group_percentage = group_count.div(group_count.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    group_percentage.plot(kind='bar', stacked=True, color=['red', 'blue'], alpha=0.7, ax=ax)

    ax.set_title(f"Segmented Bar Chart of {col1} and {col2}", color="red", size=17)
    ax.set_xlabel(col1, color="blue", size=12)
    ax.set_ylabel("Total Percentage", color="blue", size=12)

    unique_values = data[col2].unique()
    ax.legend(unique_values)

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url



# ------------------------------------------------------------------------------------ #
#                                Multivariate Analysis                                 #
# ------------------------------------------------------------------------------------ #


# .................................................... #
#                      Tree Map                        #
# .................................................... #

def treemap(df, col1, col2):
    # Drop rows with missing values in the specified columns
    df = df.dropna(subset=[col1, col2])

    # Group by the two categorical columns and calculate counts
    grouped_data = df.groupby([col1, col2]).size().reset_index(name='Count')

    # Calculate the percentage of each group
    total_count = grouped_data['Count'].sum()
    grouped_data['Percentage'] = grouped_data['Count'] / total_count * 100

    # Create labels for each section
    grouped_data['Label'] = (grouped_data[col1] + ' - ' + grouped_data[col2] + ' - ' +
                             grouped_data['Count'].astype(str) + '\n' + grouped_data['Percentage'].round(1).astype(str) + '%')

    # Define a color palette
    colors = plt.cm.get_cmap('Dark2', len(grouped_data))
    colors = [colors(i) for i in range(len(grouped_data))]

    # Plot the treemap
    fig, ax = plt.subplots(figsize=(8, 6))
    squarify.plot(
        sizes=grouped_data['Count'],
        label=grouped_data['Label'],
        alpha=0.8,
        color=colors,
        ax=ax
    )
    ax.set_title("Tree Map", color="red", size=17)
    plt.axis('off')

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url

# .................................................... #
#                     Mosaic plot                      #
# .................................................... #


def mosaic_plot(data, col1, col2):
    mosaic_data = pd.crosstab(data[col1], data[col2])

    # Calculate percentages
    mosaic_data_perc = mosaic_data.div(mosaic_data.sum(axis=1), axis=0)

    def custom_labelizer(keys):
        row, col = keys
        return f"{row}\n{mosaic_data_perc.loc[row, col]:.2%}"

    fig, ax = plt.subplots(figsize=(8, 6))
    mosaic(mosaic_data.stack(), title='', labelizer=custom_labelizer, ax=ax)
    ax.set_title('Mosaic Plot', color='red', size=17)
    ax.set_xlabel(col2)
    ax.set_ylabel(col1)

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url





# =================================================================================================== #
#                                   3. Categorical vs. Quantitative                                   #
# =================================================================================================== #


# ------------------------------------------------------------------------------------ #
#                                Univariate Analysis                                   #
# ------------------------------------------------------------------------------------ #


# .................................................... #
#                      Count plot                      #
# .................................................... #

def count_plot(data, col1):
    plt.figure(figsize=(6, 4))
    sns.countplot(data[col1], palette='Set2', edgecolor='black')

    plt.title(f"Count plot of {col1}", color="red", size=17)
    plt.xlabel(f'{col1}', color="blue", size=12)
    plt.ylabel("Count", color="blue", size=12)

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url

# .................................................... #
#                      Bar chart                       #
# .................................................... #

def bar_chart(data, col1):
    plt.figure(figsize=(6, 4))
    sns.barplot(data[col1].value_counts().values, data[col1].value_counts().index, palette='Set2', 
                edgecolor='black', orient='h')

    plt.title(f"Bar chart of {col1}", color="red", size=17)
    plt.xlabel("Count", color="blue", size=12)
    plt.ylabel(f'{col1}', color="blue", size=12)

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url

# .................................................... #
#                    Bubble chart                      #
# .................................................... #

def bubble_chart(df, col1):
    # Calculate the count of occurrences for each category
    category_counts = df[col1].value_counts().reset_index()
    category_counts.columns = [col1, 'Count']

    # Create a dictionary to map categories to numerical values
    category_map = {cat: i + 1 for i, cat in enumerate(category_counts[col1].unique())}
    category_counts['Category_num'] = category_counts[col1].map(category_map)

    # Create a bubble chart
    plt.figure(figsize=(10, 6))
    plt.scatter(
        category_counts['Category_num'], 
        category_counts['Count'], 
        s=category_counts['Count'] * 10,  # Scale the size of bubbles by count
        alpha=0.6, 
        edgecolors='w', 
        color='skyblue'
    )

    # Add labels to the bubbles
    for i in range(len(category_counts)):
        plt.text(
            category_counts['Category_num'][i], 
            category_counts['Count'][i], 
            category_counts['Count'][i], 
            ha='center', va='center', fontsize=12
        )

    # Set the ticks and labels for the x-axis
    plt.xticks(
        ticks=list(category_counts['Category_num'].unique()), 
        labels=category_counts[col1].unique()
    )
    plt.title(f'Bubble Chart of {col1}', color='red', size=17)
    plt.xlabel(col1, color='blue', size=12)
    plt.ylabel('Count', color='blue', size=12)
    plt.grid(True)

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url


# ------------------------------------------------------------------------------------ #
#                                Bivariate Analysis                                    #
# ------------------------------------------------------------------------------------ #


# .................................................... #
#                    Strip plot                        #
# .................................................... #

def strip_plot(data, col1, col2):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=data[col1], y=data[col2], color="red", edgecolor='yellow', ax=ax)
    ax.set_title(f"Strip plot of {col1} vs. {col2}", color="red", size=17)
    ax.set_xlabel(col1, color="blue", size=12)
    ax.set_ylabel(col2, color="blue", size=12)
    
    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url


# .................................................... #
#                    Density plot                      #
# .................................................... #


def density_plot(data, col1, col2):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(x=data[col1], hue=data[col2], palette="Set1", fill=True, ax=ax)
    ax.set_title(f"Kernel Density Estimation of {col1} vs. {col2}", color="red", size=17)
    ax.set_xlabel(col1, color="blue", size=12)
    ax.set_ylabel("Density", color="blue", size=12)
    
    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url


# .................................................... #
#                      Box plot                        #
# .................................................... #

def box_plot(data, col1, col2):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=data[col1], y=data[col2], palette='Set1', ax=ax)
    ax.set_title(f"Box Plot of {col1} vs. {col2}", color="red", size=17)
    ax.set_xlabel(col1, color="blue", size=12)
    ax.set_ylabel(col2, color="blue", size=12)

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url


# .................................................... #
#                    Boxen plot                        #
# .................................................... #

def boxen_plot(data, col1, col2):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxenplot(x=data[col1], y=data[col2], palette='Set1', ax=ax)
    ax.set_title(f"Boxen Plot of {col1} vs. {col2}", color="red", size=17)
    ax.set_xlabel(col1, color="blue", size=12)
    ax.set_ylabel(col2, color="blue", size=12)

    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url


# .................................................... #
#                    Violin plot                       #
# .................................................... #

def violin_plot(data, col1, col2):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(x=data[col1], y=data[col2], palette='Set1', ax=ax)
    ax.set_title(f"Violin Plot of {col1} vs. {col2}", color="red", size=17)
    ax.set_xlabel(col1, color="blue", size=12)
    ax.set_ylabel(col2, color="blue", size=12)
    
    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url


# .................................................... #
#                    Swarm plot                        #
# .................................................... #

def swarm_plot(data, col1, col2):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.swarmplot(x=data[col1], y=data[col2], ax=ax)
    ax.set_title(f"Swarm plot of {col1} vs. {col2}", color="red", size=17)
    ax.set_xlabel(col1, color="blue", size=12)
    ax.set_ylabel(col2, color="blue", size=12)
    
    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url


# ------------------------------------------------------------------------------------ #
#                                Multivariate Analysis                                 #
# ------------------------------------------------------------------------------------ #


# Define the validate_columns function
def validate_columns(data, categ_vars, quant_var):
    if len(categ_vars) < 2:
        raise ValueError("There should be at least two categorical variables.")
    
    if quant_var not in data.columns:
        raise ValueError(f"Quantitative column '{quant_var}' does not exist in the DataFrame.")
    
    for col in categ_vars:
        if col not in data.columns:
            raise ValueError(f"Categorical column '{col}' does not exist in the DataFrame.")

# Define the create_combined_column function
def create_combined_column(data, categ_vars):
    new_col_name = '_'.join(categ_vars)
    data[new_col_name] = data[categ_vars].astype(str).agg('_'.join, axis=1)
    return new_col_name

# Define the plot_categorical_vs_quantitative function
def plot_categorical_vs_quantitative(data, categ_vars, quant_var):
    # Validate columns
    validate_columns(data, categ_vars, quant_var)
    
    # Create a new combined column
    combined_col = create_combined_column(data, categ_vars)
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f'Comparison of {quant_var} by {", ".join(categ_vars)}', fontsize=16)
    
    # Define plot types and corresponding seaborn functions
    plot_types = {
        'Box Plot': sns.boxplot,
        'Bar Plot': sns.barplot,
        'Violin Plot': sns.violinplot,
        'Point Plot': sns.pointplot,
        'Swarm Plot': sns.swarmplot,
        'Strip Plot': sns.stripplot
    }
    
    # Generate plots
    for ax, (title, plot_func) in zip(axes.flatten(), plot_types.items()):
        plot_func(ax=ax, x=combined_col, y=quant_var, data=data)
        ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot to a BytesIO object
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode('utf8')}"
    plt.close()
    return plot_url
    
    # Encode the BytesIO object to a base64 string
    plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return f'data:image/png;base64,{plot_url}'

@app.route('/plot_categorical_vs_quantitative', methods=['POST'])
def plot_categorical_vs_quantitative_endpoint():
    try:
        # Extract request data
        file_path = request.form['file_path']
        categ_vars = request.form.getlist('categ_vars')
        quant_var = request.form['quant_var']
        
        # Load data
        data = pd.read_csv(file_path)
        
        # Generate plot
        plot_url = plot_categorical_vs_quantitative(data, categ_vars, quant_var)
        
        return render_template('plot.html', plot_urls=[plot_url])
    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)


