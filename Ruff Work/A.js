document.getElementById('fileInput').addEventListener('change', handleFileUpload);

let parsedData = null;

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (file) {
    Papa.parse(file, {
      header: true,
      complete: function(results) {
        parsedData = results.data;
        const columns = Object.keys(parsedData[0]);
        populateColumnsSelection(columns);
      }
    });
  }
}

function populateColumnsSelection(columns) {
  const columnsSelect = document.getElementById('columns');
  columnsSelect.innerHTML = '';
  columns.forEach(column => {
    const option = document.createElement('option');
    option.value = column;
    option.text = column;
    columnsSelect.appendChild(option);
  });
  document.getElementById('columnsSelection').classList.remove('hidden');
}

document.getElementById('plotButton').addEventListener('click', generatePlots);

function generatePlots() {
  const selectedColumns = Array.from(document.getElementById('columns').selectedOptions).map(option => option.value);
  const analysisType = document.getElementById('analysisType').value;

  if (selectedColumns.length < 2) {
    alert('Please select at least 2 columns.');
    return;
  }

  createRecommendedPlots(parsedData, selectedColumns, analysisType);
}

function createRecommendedPlots(data, columns, analysisType) {
  const plotsContainer = document.getElementById('plots');
  plotsContainer.innerHTML = '';

  if (analysisType === 'bivariate' && columns.length === 2) {
    const trace = {
      x: data.map(d => d[columns[0]]),
      y: data.map(d => d[columns[1]]),
      mode: 'markers',
      type: 'scatter'
    };

    const layout = {
      title: `Scatter Plot of ${columns[0]} vs ${columns[1]}`,
      xaxis: { title: columns[0] },
      yaxis: { title: columns[1] }
    };

    const plotDiv = document.createElement('div');
    plotDiv.id = 'plot1';
    plotsContainer.appendChild(plotDiv);

    Plotly.newPlot(plotDiv, [trace], layout);
  }

  // Additional logic for univariate and multivariate analyses
}
