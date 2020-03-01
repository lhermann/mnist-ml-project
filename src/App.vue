<template>
  <div id="app">
    <div class="d-flex align-items-center justify-content-between mb-3">
      <h1 class="mb-0">Machine Learning with the MNIST Data Set</h1>
      <b-button variant="outline-secondary" size="sm" @click="toggleVisor">Toggle Sidebar</b-button>
    </div>

    <b-card class="mb-4" title="1: Initialize Model">
      <div>
        <b-form-group label="Choose your Machine Learning Model">
          <strong>Convolutional Neural Networks</strong>
          <b-form-radio v-model="selectedModel" name="ml-model" value="tutorial">The Tutorial (2 conv. layers + 2 downsamplings)</b-form-radio>
          <b-form-radio v-model="selectedModel" name="ml-model" value="direct">The Direct (1 conv. layers + 1 downsampling)</b-form-radio>
          <b-form-radio v-model="selectedModel" name="ml-model" value="complex">The Complex (3 conv. layers)</b-form-radio>
          <b-form-radio v-model="selectedModel" name="ml-model" value="simple">The Simple (2 downsamplings + 1 conv. layers)</b-form-radio>
          <strong>Dense Neural Networks</strong>
          <b-form-radio v-model="selectedModel" name="ml-model" value="huge">The Huge (no downsampling + 2 dense layers)</b-form-radio>
          <b-form-radio v-model="selectedModel" name="ml-model" value="large">The Large (1 downsampling + 2 dense layers)</b-form-radio>
          <b-form-radio v-model="selectedModel" name="ml-model" value="modest">The Modest (no downsampling + 1 dense layers)</b-form-radio>
          <b-form-radio v-model="selectedModel" name="ml-model" value="small">The Small (1 downsampling + 1 dense layers)</b-form-radio>
          <b-form-radio v-model="selectedModel" name="ml-model" value="tiny">The Tiny (2 downsamplings + 1 dense layers)</b-form-radio>
        </b-form-group>
      </div>
      <b-alert :show="modelSummary" variant="secondary">
        <code>
          <pre>{{ modelSummary }}</pre>
        </code>
      </b-alert>
      <template v-slot:footer>
        <div class="d-flex align-items-center justify-content-between">
          <div>
            <b-button
              variant="primary"
              :disabled="steps.trainModel === 'done'"
              @click="initModel"
            >
              Initialize Model
            </b-button>
            &nbsp;
            <b-button
              variant="outline-secondary"
              :disabled="steps.trainModel !== 'done'"
              @click="resetModel"
            >
              Reset Model
            </b-button>
          </div>
          <b-spinner v-show="steps.initModel === 'pending'" variant="success"></b-spinner>
          <b-icon v-show="steps.initModel === 'done'" icon="check" variant="success" font-scale="2"/>
        </div>
        <b-alert v-if="errors.initModel" class="mt-3" variant="danger" show>{{ errors.initModel }}</b-alert>
      </template>
    </b-card>

    <b-card class="mb-4" title="2: Load Data">
      <div class="text-muted">Load in all image and label data sets from MNIST</div>

      <div ref="loadDataExamples"></div>

      <template v-slot:footer>
        <div class="d-flex align-items-center justify-content-between">
          <b-button variant="primary" :disabled="steps.loadData === 'done'" @click="loadData">
            Load Data
          </b-button>
          <b-spinner v-show="steps.loadData === 'pending'" variant="success"></b-spinner>
          <b-icon v-show="steps.loadData === 'done'" icon="check" variant="success" font-scale="2"/>
        </div>
        <b-alert v-if="errors.loadData" class="mt-3" variant="danger" show>{{ errors.loadData }}</b-alert>
      </template>
    </b-card>

    <b-card class="mb-4" title="3: Train Model">
      <b-row>
        <b-col cols="6">
          <b-form-group label="Batch Size">
            <b-form-input v-model="train.batchSize" number type="number" />
          </b-form-group>
        </b-col>
        <b-col cols="6">
          <b-form-group label="Epochs">
            <b-form-input v-model="train.epochs" number type="number" />
          </b-form-group>
        </b-col>
        <b-col cols="6">
          <b-form-group label="Training Data Size">
            <b-form-input v-model="train.trainDataSize" number type="number" />
          </b-form-group>
        </b-col>
        <b-col cols="6">
          <b-form-group label="Validation Data Size">
            <b-form-input v-model="train.valDataSize" number type="number" />
          </b-form-group>
        </b-col>
      </b-row>
      <div>
      </div>

      <template v-slot:footer>
        <div class="d-flex align-items-center justify-content-between">
          <div>
            <b-button
              variant="primary"
              :disabled="steps.loadData !== 'done'"
              @click="trainModel"
            >
              Train Model
            </b-button>
            &nbsp;
            <b-button
              variant="outline-primary"
              :disabled="steps.loadData !== 'done'"
              @click="trainModel({ background: true })"
            >
              Train in background
            </b-button>
          </div>
          <b-spinner v-show="steps.trainModel === 'pending'" variant="success"></b-spinner>
          <b-icon v-show="steps.trainModel === 'done'" icon="check" variant="success" font-scale="2"/>
        </div>
        <b-alert v-if="errors.trainModel" class="mt-3" variant="danger" show>{{ errors.trainModel }}</b-alert>
      </template>
    </b-card>

    <b-card class="mb-4" title="4: Analyze Model">
      <p class="text-muted">Analyze the model with 1000 images from the validation set. A class accuracy table and confusion matrix are generated in the sidebar.</p>

      <p>
        <span>Accuracy: </span>
        <strong v-if="accuracy">{{ accuracy | percent }}%</strong>
        <span v-else>&mdash; data not yet analyzed &mdash;</span>
      </p>

      <template v-slot:footer>
        <div class="d-flex align-items-center justify-content-between">
          <b-button
            variant="primary"
            :disabled="steps.loadData !== 'done'"
            @click="analyze"
          >
            Analyze Model
          </b-button>
          <b-spinner v-show="steps.analyze === 'pending'" variant="success"></b-spinner>
          <b-icon v-show="steps.analyze === 'done'" icon="check" variant="success" font-scale="2"/>
        </div>
        <b-alert v-if="errors.analyze" class="mt-3" variant="danger" show>{{ errors.analyze }}</b-alert>
      </template>
    </b-card>

    <b-card class="mb-4" title="5: Predict">
      <h5>Example Data</h5>
      <div v-if="!exampleTensors.length" class="text-muted">&mdash; no examples yet &mdash;</div>
      <div ref="exampleCanvasHolder" class="d-flex justify-content-between"></div>
      <div class="d-flex justify-content-between mt-2">
        <div v-for="tensor in exampleTensors">
          <b-button variant="outline-primary" size="sm" @click="predict(tensor)">
            <b-icon icon="check-circle" scale="1.2"/>
          </b-button>
        </div>
      </div>

      <h5 class="mt-4">Prediction Result</h5>
      <div v-if="predictionResult" class="d-flex">
        <div
          v-for="(probability, index) in predictionResult"
          class="text-center p-2 flex-grow-1"
          :style="{ 'background-color': `rgba(94, 186, 125, ${probability})` }"
        >
          <div class="font-weight-bold">{{ index }}</div>
          <div>{{ probability | percent }}%</div>
        </div>
      </div>
      <div v-else class="text-muted">&mdash; no result yet &mdash;</div>

      <template v-slot:footer>
        <b-button variant="outline-primary" :disabled="steps.loadData !== 'done'" @click="prepareExampleTensors">
          Refresh
        </b-button>
        <b-alert v-if="errors.predict" class="mt-3" variant="danger" show>{{ errors.predict }}</b-alert>
      </template>
    </b-card>

    <p class="text-muted">A tool by <a href="https://tentmaker.dev/" target="_blank">Lukas Hermann</a></p>
  </div>
</template>

<script>
import MnistData from './utils/MnistData.js';
import Model from './utils/Model.js';
import Analyzer from './utils/Analyzer.js';

const data = new MnistData(tf)
const model = new Model(tf, tfvis, data)
const analyzer = new Analyzer(tfvis, model)

export default {
  name: 'App',
  data () {
    return {
      errors: {
        initModel: null,
        loadData: null,
        trainModel: null,
        analyze: null,
        predict: null,
      },
      steps: {
        loadData: false,
        initModel: false,
        trainModel: false,
        analyze: false,
        predict: false,
      },
      selectedModel: 'tutorial',
      modelSummary: null,
      train: {
        batchSize: 500,
        epochs: 10,
        trainDataSize: 10000,
        valDataSize: 2000,
      },
      accuracy: null,
      exampleTensors: [],
      predictionResult: null,
      tfvisVisor: null
    }
  },
  watch: {
    selectedModel (value) {
      this.setUrlParam('selectedModel', value)
    },
    train: {
      handler (obj) {
        for (const key in obj) {
          this.setUrlParam(key, obj[key])
        }
      },
      deep: true
    }
  },
  mounted () {
    window.data = data
    window.model = model
    window.analyzer = analyzer
    this.tfvisVisor = tfvis.visor()
    this.selectedModel = this.getUrlParam('selectedModel') || this.selectedModel
    this.train.batchSize = this.getUrlParam('batchSize') || this.train.batchSize
    this.train.epochs = this.getUrlParam('epochs') || this.train.epochs
    this.train.trainDataSize = this.getUrlParam('trainDataSize') || this.train.trainDataSize
    this.train.valDataSize = this.getUrlParam('valDataSize') || this.train.valDataSize
  },
  methods: {
    async initModel () {
      this.steps.initModel = 'pending'
      this.errors.initModel = null
      try {
        model.initModel(this.selectedModel)
        this.modelSummary = model.getSummary()
      } catch (e) {
        this.errors.initModel = e
      }
      this.steps.initModel = 'done'
    },
    resetModel () {
      model.resetModel()
      this.steps.initModel = false
      this.steps.trainModel = false
      this.modelSummary = ''
    },
    async loadData () {
      this.steps.loadData = 'pending'
      this.errors.loadData = null
      try {
        await data.load();
        await data.renderExampleImages(this.$refs.loadDataExamples, 38)
        this.prepareExampleTensors()
      } catch (e) {
        this.errors.loadData = e
      }
      this.steps.loadData = 'done'
    },
    async trainModel ({ background }) {
      this.steps.trainModel = 'pending'
      this.errors.trainModel = null
      try {
        await model.train(
          parseInt(this.train.batchSize),
          parseInt(this.train.epochs),
          parseInt(this.train.trainDataSize),
          parseInt(this.train.valDataSize),
          background
        );
      } catch (error) {
        this.errors.trainModel = error
      }
      this.steps.trainModel = 'done'
    },
    async analyze () {
      this.steps.analyze = 'pending'
      this.errors.analyze = null
      try {
        const [predictions, labels] = model.doPrediction(1000)
        await analyzer.showAccuracy(predictions, labels);
        await analyzer.showConfusion(predictions, labels);
        this.accuracy = await tfvis.metrics.accuracy(labels, predictions);
        predictions.dispose();
        labels.dispose();
      } catch (e) {
        this.errors.analyze = e
      }
      this.steps.analyze = 'done'
    },
    async predict (tensor) {
      this.steps.predict = 'pending'
      this.errors.predict = null
      try {
        this.predictionResult = model.doSinglePrediction(tensor)
      } catch (error) {
        this.errors.predict = error
      }
      this.steps.predict = 'done'
    },
    async prepareExampleTensors () {
      this.exampleTensors = await data.renderExampleImages(this.$refs.exampleCanvasHolder, 12)
    },
    toggleVisor () {
      this.tfvisVisor.toggle()
    },
    setUrlParam (key, value) {
      if (history.pushState) {
        const url = new URL(window.location.href)
        url.searchParams.set(key, value)
        window.history.pushState(null, null, url.href)
      }
    },
    getUrlParam (key) {
      const url = new URL(window.location.href)
      return url.searchParams.get(key)
    }
  }
}
</script>

<style>
body {
  background-color: #eee !important;
}
#app {
  padding: 2rem;
  max-width: 800px;
}
.btn.disabled,
.btn:disabled {
  cursor: not-allowed;
}
</style>
