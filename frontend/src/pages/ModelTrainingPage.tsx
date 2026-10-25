import React, { useEffect, useState } from "react";
import {
  Cpu,
  Play,
  RotateCw,
  CheckCircle,
  AlertTriangle,
  FileBarChart,
  HardDrive,
  Download,
  Gauge,
  Database,
} from "lucide-react";
import {
  fetchModelStatus,
  fetchTrainingStatus,
  startTraining,
  evaluateModel,
  fetchMetrics,
  fetchDatasetStatus,
  prepareDataset,
  BACKEND_URL,
} from "../services/api";
import { ModelMetricsData } from "../types";

export const ModelTrainingPage: React.FC = () => {
  const [modelStatus, setModelStatus] = useState<any>(null);
  const [trainingStatus, setTrainingStatus] = useState<any>(null);
  const [metrics, setMetrics] = useState<ModelMetricsData | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<any>(null);

  // Training form state
  const [epochs, setEpochs] = useState<number>(10);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [learningRate, setLearningRate] = useState<number>(0.0001);
  const [useAugmentation, setUseAugmentation] = useState<boolean>(true);

  const [isStartingTrain, setIsStartingTrain] = useState<boolean>(false);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [isPreparingData, setIsPreparingData] = useState<boolean>(false);
  const [exportMessage, setExportMessage] = useState<string | null>(null);

  const loadData = () => {
    fetchModelStatus().then(setModelStatus).catch(() => {});
    fetchTrainingStatus().then(setTrainingStatus).catch(() => {});
    fetchDatasetStatus().then(setDatasetStatus).catch(() => {});
    fetchMetrics()
      .then((res) => {
        if (res.available && res.metrics) setMetrics(res.metrics);
      })
      .catch(() => {});
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(() => {
      fetchTrainingStatus()
        .then((ts) => {
          setTrainingStatus(ts);
          if (ts && ts.is_training) {
            // keep polling training status
          }
        })
        .catch(() => {});
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleStartTraining = async () => {
    setIsStartingTrain(true);
    try {
      await startTraining(epochs, batchSize, learningRate, useAugmentation);
      loadData();
    } catch (e: any) {
      alert(`Training trigger error: ${e.message}`);
    } finally {
      setIsStartingTrain(false);
    }
  };

  const handleRunEvaluation = async () => {
    setIsEvaluating(true);
    try {
      const res = await evaluateModel();
      if (res && res.test_accuracy !== undefined) {
        setMetrics(res);
      }
    } catch (e: any) {
      alert(`Evaluation error: ${e.message}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  const handlePrepareDataset = async () => {
    setIsPreparingData(true);
    try {
      await prepareDataset();
      loadData();
    } catch (e: any) {
      alert(`Dataset preparation error: ${e.message}`);
    } finally {
      setIsPreparingData(false);
    }
  };

  return (
    <div className="p-6 max-w-[1700px] mx-auto space-y-6">
      {/* Top Banner / Dataset & Checkpoint Readiness */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Dataset Status */}
        <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm space-y-3">
          <div className="flex items-center justify-between border-b border-fleet-border pb-2">
            <span className="text-[14px] font-bold text-fleet-text flex items-center gap-1.5">
              <Database className="w-4 h-4 text-fleet-primary" />
              Dataset Status & Partitioning
            </span>
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-bold uppercase ${
                datasetStatus?.processed_splits.ready
                  ? "bg-[#F0F8F3] text-fleet-normal border border-[#D1E8D8]"
                  : "bg-[#FFF9F0] text-fleet-warning border border-[#FCE6C7]"
              }`}
            >
              {datasetStatus?.processed_splits.ready ? "Partitions Ready" : "Partitions Missing"}
            </span>
          </div>

          <div className="text-xs space-y-1.5 text-fleet-muted">
            <div>
              <span className="font-semibold text-fleet-text">Train Samples:</span>{" "}
              {datasetStatus?.processed_splits.counts.train || 0} images (Subjects 1-8)
            </div>
            <div>
              <span className="font-semibold text-fleet-text">Val Samples:</span>{" "}
              {datasetStatus?.processed_splits.counts.val || 0} images (Subjects 9-10)
            </div>
            <div>
              <span className="font-semibold text-fleet-text">Held-Out Test:</span>{" "}
              {datasetStatus?.processed_splits.counts.test || 0} images (Subjects 11-12)
            </div>
            <div className="text-[11px] text-fleet-muted italic pt-1">
              Zero subject leakage across partitions. Supports NTHU-DDD & UL-DD drop-in.
            </div>
          </div>

          {!datasetStatus?.processed_splits.ready && (
            <button
              onClick={handlePrepareDataset}
              disabled={isPreparingData}
              className="w-full flex items-center justify-center gap-1.5 py-1.5 text-xs font-semibold bg-fleet-primary text-white rounded hover:bg-fleet-primaryHover transition-colors disabled:opacity-50"
            >
              <RotateCw className={`w-3.5 h-3.5 ${isPreparingData ? "animate-spin" : ""}`} />
              <span>{isPreparingData ? "Preparing Partitions..." : "Generate Benchmark Partitions"}</span>
            </button>
          )}
        </div>

        {/* Model Checkpoint Status */}
        <div className="bg-white border border-fleet-border rounded-lg p-4 shadow-sm space-y-3">
          <div className="flex items-center justify-between border-b border-fleet-border pb-2">
            <span className="text-[14px] font-bold text-fleet-text flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-fleet-primary" />
              Active Checkpoint
            </span>
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-bold uppercase ${
                modelStatus?.is_loaded
                  ? "bg-[#F0F8F3] text-fleet-normal border border-[#D1E8D8]"
                  : "bg-[#FDF2F2] text-fleet-critical border border-[#F8D2D2]"
              }`}
            >
              {modelStatus?.is_loaded ? "Checkpoint Loaded" : "No Weights Loaded"}
            </span>
          </div>

          <div className="text-xs space-y-1.5 text-fleet-muted">
            <div>
              <span className="font-semibold text-fleet-text">Architecture:</span> MobileNetV3-Small
            </div>
            <div>
              <span className="font-semibold text-fleet-text">Checkpoint Path:</span>{" "}
              <code className="text-[11px] bg-fleet-bg px-1 py-0.5 rounded">
                checkpoints/best_mobilenet_v3.pth
              </code>
            </div>
            <div>
              <span className="font-semibold text-fleet-text">Inference Hardware:</span>{" "}
              {modelStatus?.device || "CPU"}
            </div>
            <div className="text-[11px] text-fleet-muted">
              Val Accuracy: {modelStatus?.metadata?.val_acc ? `${(modelStatus.metadata.val_acc * 100).toFixed(1)}%` : "N/A"}
            </div>
          </div>

          <button
            onClick={handleRunEvaluation}
            disabled={isEvaluating || !modelStatus?.checkpoint_exists}
            className="w-full flex items-center justify-center gap-1.5 py-1.5 text-xs font-semibold bg-white border border-fleet-border text-fleet-text rounded hover:bg-fleet-bg transition-colors disabled:opacity-50"
          >
            <FileBarChart className={`w-3.5 h-3.5 ${isEvaluating ? "animate-spin" : ""}`} />
            <span>{isEvaluating ? "Evaluating on Test Set..." : "Run Test Set Evaluation"}</span>
          </button>
        </div>
      </div>

      {/* Training Configuration & Execution Panel */}
      <div className="bg-white border border-fleet-border rounded-lg p-5 shadow-sm space-y-4">
        <h2 className="text-[15px] font-bold text-fleet-text border-b border-fleet-border pb-2 m-0">
          Transfer Learning Training Pipeline (MobileNetV3)
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="text-xs font-semibold text-fleet-muted block mb-1">
              Training Epochs
            </label>
            <input
              type="number"
              min="1"
              max="50"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
              className="w-full text-xs border border-fleet-border rounded px-3 py-1.5 bg-fleet-bg focus:ring-1 focus:ring-fleet-primary"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-fleet-muted block mb-1">
              Batch Size
            </label>
            <input
              type="number"
              min="4"
              max="64"
              step="4"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 16)}
              className="w-full text-xs border border-fleet-border rounded px-3 py-1.5 bg-fleet-bg focus:ring-1 focus:ring-fleet-primary"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-fleet-muted block mb-1">
              Learning Rate (AdamW)
            </label>
            <input
              type="number"
              step="0.00005"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.0001)}
              className="w-full text-xs border border-fleet-border rounded px-3 py-1.5 bg-fleet-bg focus:ring-1 focus:ring-fleet-primary"
            />
          </div>

          <div className="flex items-center pt-5">
            <label className="flex items-center gap-2 text-xs font-semibold text-fleet-text cursor-pointer">
              <input
                type="checkbox"
                checked={useAugmentation}
                onChange={(e) => setUseAugmentation(e.target.checked)}
                className="rounded border-fleet-border text-fleet-primary focus:ring-0"
              />
              <span>Data Augmentation (Flip, Color, Rotation)</span>
            </label>
          </div>
        </div>

        {/* Start Training Button */}
        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={handleStartTraining}
            disabled={trainingStatus?.is_training || isStartingTrain}
            className="flex items-center gap-1.5 px-4 py-2 bg-fleet-primary hover:bg-fleet-primaryHover text-white rounded text-xs font-bold transition-colors disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-white" />
            <span>
              {trainingStatus?.is_training
                ? `Training in Progress (Epoch ${trainingStatus.current_epoch}/${trainingStatus.total_epochs})...`
                : "Start MobileNetV3 Training"}
            </span>
          </button>
        </div>

        {/* Live Training Progress Log */}
        {trainingStatus?.is_training && (
          <div className="bg-[#1B222C] text-gray-200 p-3.5 rounded-lg text-xs font-mono space-y-1 max-h-48 overflow-y-auto">
            <div className="text-fleet-warning font-semibold">
              Status: {trainingStatus.status} | Epoch {trainingStatus.current_epoch} of {trainingStatus.total_epochs}
            </div>
            <div>
              Train Loss: {trainingStatus.train_loss} | Train Acc: {(trainingStatus.train_acc * 100).toFixed(1)}% | Val Loss: {trainingStatus.val_loss} | Val Acc: {(trainingStatus.val_acc * 100).toFixed(1)}%
            </div>
          </div>
        )}
      </div>

      {/* Model Evaluation & Analytics Results */}
      {metrics && (
        <div className="bg-white border border-fleet-border rounded-lg p-5 shadow-sm space-y-5">
          <div className="flex flex-wrap items-center justify-between border-b border-fleet-border pb-3">
            <div>
              <h2 className="text-[15px] font-bold text-fleet-text m-0">
                Held-Out Test Set Evaluation Results
              </h2>
              <p className="text-xs text-fleet-muted mt-0.5">
                Evaluated on {metrics.total_test_samples} untouched test samples (Device: {metrics.device_tested})
              </p>
            </div>
            <div className="text-xs font-mono text-fleet-muted">
              Latency: Mean {metrics.latency_benchmarks_ms.mean}ms • P95 {metrics.latency_benchmarks_ms.p95}ms
            </div>
          </div>

          {/* Metric Tiles */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-fleet-bg border border-fleet-border rounded p-3 text-center">
              <div className="text-[11px] font-bold text-fleet-muted uppercase">
                Test Accuracy
              </div>
              <div className="text-xl font-bold font-mono text-fleet-normal mt-1">
                {(metrics.test_accuracy * 100).toFixed(1)}%
              </div>
            </div>

            <div className="bg-fleet-bg border border-fleet-border rounded p-3 text-center">
              <div className="text-[11px] font-bold text-fleet-muted uppercase">
                Precision
              </div>
              <div className="text-xl font-bold font-mono text-fleet-primary mt-1">
                {(metrics.precision * 100).toFixed(1)}%
              </div>
            </div>

            <div className="bg-fleet-bg border border-fleet-border rounded p-3 text-center">
              <div className="text-[11px] font-bold text-fleet-muted uppercase">
                Recall
              </div>
              <div className="text-xl font-bold font-mono text-fleet-primary mt-1">
                {(metrics.recall * 100).toFixed(1)}%
              </div>
            </div>

            <div className="bg-fleet-bg border border-fleet-border rounded p-3 text-center">
              <div className="text-[11px] font-bold text-fleet-muted uppercase">
                F1-Score
              </div>
              <div className="text-xl font-bold font-mono text-fleet-primary mt-1">
                {(metrics.f1_score * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Genuine Plot Images Generated by evaluate.py */}
          <div>
            <div className="text-xs font-bold text-fleet-text uppercase tracking-wider mb-2">
              Generated Evaluation Curves & Diagnostic Charts:
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="border border-fleet-border rounded-lg overflow-hidden bg-fleet-bg">
                <div className="bg-white px-3 py-1.5 border-b border-fleet-border text-xs font-semibold text-fleet-text">
                  Confusion Matrix (Test Set)
                </div>
                <img
                  src={`${BACKEND_URL}/plots/confusion_matrix.png?t=${metrics.evaluation_timestamp}`}
                  alt="Confusion Matrix"
                  className="w-full object-contain max-h-72"
                />
              </div>

              <div className="border border-fleet-border rounded-lg overflow-hidden bg-fleet-bg">
                <div className="bg-white px-3 py-1.5 border-b border-fleet-border text-xs font-semibold text-fleet-text">
                  ROC Curve (AUC: {metrics.auc_roc})
                </div>
                <img
                  src={`${BACKEND_URL}/plots/roc_curve.png?t=${metrics.evaluation_timestamp}`}
                  alt="ROC Curve"
                  className="w-full object-contain max-h-72"
                />
              </div>

              <div className="border border-fleet-border rounded-lg overflow-hidden bg-fleet-bg">
                <div className="bg-white px-3 py-1.5 border-b border-fleet-border text-xs font-semibold text-fleet-text">
                  Training & Validation Accuracy History
                </div>
                <img
                  src={`${BACKEND_URL}/plots/accuracy_curves.png?t=${metrics.evaluation_timestamp}`}
                  alt="Accuracy Curves"
                  className="w-full object-contain max-h-72"
                />
              </div>

              <div className="border border-fleet-border rounded-lg overflow-hidden bg-fleet-bg">
                <div className="bg-white px-3 py-1.5 border-b border-fleet-border text-xs font-semibold text-fleet-text">
                  Training & Validation Loss History
                </div>
                <img
                  src={`${BACKEND_URL}/plots/loss_curves.png?t=${metrics.evaluation_timestamp}`}
                  alt="Loss Curves"
                  className="w-full object-contain max-h-72"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
