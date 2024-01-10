require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

/**
 * k-nearest algorithm. 
 */
function knn(features, labels, predictionPoint, k) {
    const { mean, variance } = tf.moments(features, 0);
    const scaledPredication = predictionPoint.sub(mean).div(variance.pow(0.5));

    return features
        .sub(mean) // standardize.
        .div(variance.pow(0.5))
        .sub(scaledPredication) // Knn distance calc.
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1) // Knn other steps.
        .concat(labels, 1)
        .unstack()
        .sort((a,b) => a.get(0) > b.get(0) ? 1 : -1)
        .slice(0, k)
        .reduce((acc, pair) => acc + pair.get(1), 0) / k;
}

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

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
    const result = knn(features, labels, tf.tensor(testPoint), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0];
    console.log('Error', err * 100)
})
