const CLASS_NAMES = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

export default class Analyzer {
  constructor(tfvis, model) {
    this.tfvis = tfvis
    this.model = model;
  }

  async showAccuracy(preds, labels) {
    const classAccuracy = await this.tfvis.metrics.perClassAccuracy(labels, preds);
    const container = {name: 'Accuracy', tab: 'Evaluation'};
    this.tfvis.show.perClassAccuracy(container, classAccuracy, CLASS_NAMES);
  }

  async showConfusion(preds, labels) {
    const confusionMatrix = await this.tfvis.metrics.confusionMatrix(labels, preds);
    const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
    this.tfvis.render.confusionMatrix(container, {values: confusionMatrix}, CLASS_NAMES);
  }
}
