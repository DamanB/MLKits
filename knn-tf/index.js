require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

let { 
    features, 
    labels, 
    testFeatures, 
    testLabels 
} = loadCSV('kc_house_data.csv', {
    shuffle: true, // For getting test points, we should shuffle our data and select.
    splitTest: 10,
    dataColumns: ['lat', 'long'],
    labelColumns: ['price'],
});

console.log(    
    testFeatures, 
    testLabels 
);