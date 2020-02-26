const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const IMAGE_CHANNELS = 1;

export default class Model {
  constructor(tensorflow, tensorflowVis, data) {
    this.tf = tensorflow
    this.tfvis = tensorflowVis
    this.model = tensorflow.sequential();
    this.data = data
  }

  initModel (modelArchitecture = 'default') {
    this.initDefaultModel()
    this.tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Architecture'}, this.model);
  }

  getSummary () {
    let str = ''
    this.model.summary(undefined, undefined, line => str += line + "\n")
    return str
  }

  initDefaultModel() {
    // In the first layer of our convolutional neural network we have
    // to specify the input shape. Then we specify some parameters for
    // the convolution operation that takes place in this layer.
    this.model.add(this.tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));

    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.
    this.model.add(this.tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Repeat another conv2d + maxPooling stack.
    // Note that we have more filters in the convolution.
    this.model.add(this.tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    this.model.add(this.tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    this.model.add(this.tf.layers.flatten());

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    this.model.add(this.tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));


    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = this.tf.train.adam();
    this.model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  }

  async train (batchSize = 512, trainDataSize = 5500, testDataSize = 1000) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', styles: { height: '1000px' }, tab: 'Training'
    };
    const fitCallbacks = this.tfvis.show.fitCallbacks(container, metrics);

    const [trainXs, trainYs] = this.tf.tidy(() => {
      const d = this.data.nextTrainBatch(trainDataSize);
      return [
        d.xs.reshape([trainDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]),
        d.labels
      ];
    });

    const [testXs, testYs] = this.tf.tidy(() => {
      const d = this.data.nextTestBatch(trainDataSize);
      return [
        d.xs.reshape([trainDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]),
        d.labels
      ];
    });

    return this.model.fit(trainXs, trainYs, {
      batchSize,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

  doPrediction(testDataSize = 500) {
    const testData = this.data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax([-1]);
    const preds = this.model.predict(testxs).argMax([-1]);

    testxs.dispose();
    return [preds, labels];
  }

  doSinglePrediction(tensor) {
    const testX = tensor.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    const resTensor = this.model.predict(testX)
    return resTensor.arraySync()[0]
  }
}
