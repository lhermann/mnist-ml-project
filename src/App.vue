<template>
  <div id="app">
    <b-card class="mb-4" title="1: Load Data">
      <template v-slot:footer>
        <div class="d-flex align-items-center justify-content-between">
          <b-button variant="primary" :disabled="steps.loadData !== false" @click="loadData">
            Load Data
          </b-button>
          <b-spinner v-show="steps.loadData === 'pending'" variant="success"></b-spinner>
          <b-icon v-show="steps.loadData === 'done'" icon="check" variant="success" font-scale="2"/>
        </div>
      </template>
    </b-card>

    <b-card class="mb-4" title="2: Initialize Model">
      <b-alert :show="modelSummary" variant="secondary">
        <code>
          <pre>{{ modelSummary }}</pre>
        </code>
      </b-alert>
      <template v-slot:footer>
        <div class="d-flex align-items-center justify-content-between">
          <b-button
            variant="primary"
            :disabled="steps.loadData !== 'done'"
            @click="initModel"
          >
            Initialize Model
          </b-button>
          <b-spinner v-show="steps.initModel === 'pending'" variant="success"></b-spinner>
          <b-icon v-show="steps.initModel === 'done'" icon="check" variant="success" font-scale="2"/>
        </div>
      </template>
    </b-card>

    <b-card class="mb-4" title="3: Teach Model">
      <div>
        <b-form-group label="Batch Size">
          <b-form-input v-model="train.batchSize" number type="number" />
        </b-form-group>
        <b-form-group label="Train Data Size">
          <b-form-input v-model="train.trainDataSize" number type="number" />
        </b-form-group>
        <b-form-group label="Test Data Size">
          <b-form-input v-model="train.testDataSize" number type="number" />
        </b-form-group>
      </div>

      <template v-slot:footer>
        <div class="d-flex align-items-center justify-content-between">
          <b-button
            variant="primary"
            :disabled="steps.loadData !== 'done'"
            @click="teachModel"
          >
            Teach Model
          </b-button>
          <b-icon v-show="steps.teachModel === 'pending'" icon="arrow-clockwise" variant="success" font-scale="2"></b-icon>
          <b-icon v-show="steps.teachModel === 'done'" icon="check" variant="success" font-scale="2"/>
        </div>
        <b-alert v-if="errors.teachModel" class="mt-3" variant="danger" show>{{ errors.teachModel }}</b-alert>
      </template>
    </b-card>

    <b-card class="mb-4" title="4: Analyze Model">
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
  </div>
</template>

<script>
import MnistData from './utils/MnistData.js';
import showExamples from './utils/showExamples.js';
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
        teachModel: null,
        predict: null,
      },
      steps: {
        loadData: false,
        initModel: false,
        teachModel: false,
        analyze: false,
        predict: false,
      },
      modelSummary: null,
      train: {
        batchSize: 128,
        trainDataSize: 1000,
        testDataSize: 200,
      },
      exampleTensors: [],
      predictionResult: null,
    }
  },
  mounted () {
    window.data = data
    window.model = model
    window.analyzer = analyzer
    tfvis.visor()
  },
  methods: {
    async loadData () {
      this.steps.loadData = 'pending'
      await data.load();
      await showExamples(tfvis, data);
      this.prepareExampleTensors()
      this.steps.loadData = 'done'
    },
    async initModel () {
      this.steps.initModel = 'pending'
      model.initModel('default')
      this.modelSummary = model.getSummary()
      this.steps.initModel = 'done'
    },
    async teachModel () {
      this.steps.teachModel = 'pending'
      this.errors.teachModel = null
      try {
        await model.train(this.train.batchSize, this.train.trainDataSize, this.train.testDataSize);
      } catch (error) {
        this.errors.teachModel = error
      }
      this.steps.teachModel = 'done'
    },
    async analyze () {
      this.steps.analyze = 'pending'
      await analyzer.showAccuracy();
      await analyzer.showConfusion();
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
      const COUNT = 12
      const tensors = data.nextTestBatch(COUNT)
      this.$refs.exampleCanvasHolder.innerHTML = '';
      this.exampleTensors = await Promise.all(
        [...Array(COUNT).keys()].map(async (item, index) => {
          const tensor = tf.tidy(() => tensors.xs.slice([index, 0], [1, tensors.xs.shape[1]]))
          const canvas = document.createElement('canvas');
          canvas.width = 28;
          canvas.height = 28;
          canvas.style = 'margin: 4px;';
          await tf.browser.toPixels(tensor.reshape([28, 28, 1]), canvas)
          this.$refs.exampleCanvasHolder.appendChild(canvas);
          return tensor
        })
      )
    },
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
