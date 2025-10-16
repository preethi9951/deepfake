import { useState, useEffect } from "react";
import "@/App.css";
import axios from "axios";
import { Upload, AlertCircle, CheckCircle, XCircle, TrendingUp, BarChart3, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);

  useEffect(() => {
    checkModelStatus();
    fetchHistory();
    fetchStats();
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await axios.get(`${API}/model-status`);
      setModelStatus(response.data);
      if (!response.data.loaded) {
        toast.error("Model not loaded. Please train the model first.");
      }
    } catch (error) {
      console.error("Error checking model status:", error);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API}/predictions?limit=10`);
      setHistory(response.data);
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error("Error fetching stats:", error);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        toast.error("Please select a valid image file");
        return;
      }
      setSelectedFile(file);
      setResult(null);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      toast.error("Please select an image first");
      return;
    }

    if (!modelStatus?.loaded) {
      toast.error("Model is not loaded. Please run train.py first.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setResult(response.data);
      toast.success("Analysis complete!");
      fetchHistory();
      fetchStats();
    } catch (error) {
      console.error("Prediction error:", error);
      toast.error(error.response?.data?.detail || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="app-container">
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <div className="hero-badge" data-testid="hero-badge">
            <span className="badge-dot"></span>
            AI-Powered Detection
          </div>
          <h1 className="hero-title" data-testid="hero-title">
            Deepfake Image Detection
          </h1>
          <p className="hero-subtitle" data-testid="hero-subtitle">
            Advanced CNN-based system using Xception architecture to identify manipulated images
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="content-grid">
          {/* Upload Section */}
          <div className="upload-section">
            <Card className="upload-card" data-testid="upload-card">
              <CardHeader>
                <CardTitle>Upload Image</CardTitle>
                <CardDescription>Select an image to analyze for deepfake detection</CardDescription>
              </CardHeader>
              <CardContent>
                {!preview ? (
                  <label className="upload-area" data-testid="upload-area">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileSelect}
                      className="file-input"
                      data-testid="file-input"
                    />
                    <div className="upload-content">
                      <Upload className="upload-icon" />
                      <p className="upload-text">Click to upload or drag and drop</p>
                      <p className="upload-hint">PNG, JPG, JPEG up to 10MB</p>
                    </div>
                  </label>
                ) : (
                  <div className="preview-container" data-testid="preview-container">
                    <img src={preview} alt="Preview" className="preview-image" />
                    <Button
                      onClick={resetAnalysis}
                      variant="outline"
                      className="reset-btn"
                      data-testid="reset-btn"
                    >
                      Choose Different Image
                    </Button>
                  </div>
                )}

                {preview && (
                  <Button
                    onClick={handlePredict}
                    disabled={loading}
                    className="analyze-btn"
                    data-testid="analyze-btn"
                  >
                    {loading ? "Analyzing..." : "Analyze Image"}
                  </Button>
                )}
              </CardContent>
            </Card>

            {/* Result Section */}
            {result && (
              <Card className="result-card" data-testid="result-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    {result.label === "FAKE" ? (
                      <XCircle className="text-red-500" data-testid="fake-icon" />
                    ) : (
                      <CheckCircle className="text-green-500" data-testid="real-icon" />
                    )}
                    Detection Result
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="result-content">
                    <div className="result-label-section">
                      <Badge
                        className={result.label === "FAKE" ? "badge-fake" : "badge-real"}
                        data-testid="result-badge"
                      >
                        {result.label}
                      </Badge>
                      <p className="confidence-text" data-testid="confidence-text">
                        {result.confidence.toFixed(2)}% Confidence
                      </p>
                    </div>
                    <Progress
                      value={result.confidence}
                      className="confidence-progress"
                      data-testid="confidence-progress"
                    />
                    <p className="result-description" data-testid="result-description">
                      {result.label === "FAKE"
                        ? "This image shows signs of manipulation and is likely a deepfake."
                        : "This image appears to be authentic with no signs of manipulation."}
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Sidebar */}
          <div className="sidebar">
            {/* Stats Card */}
            {stats && (
              <Card className="stats-card" data-testid="stats-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    Statistics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="stats-grid">
                    <div className="stat-item" data-testid="stat-total">
                      <TrendingUp className="stat-icon" />
                      <div>
                        <p className="stat-value">{stats.total_predictions}</p>
                        <p className="stat-label">Total Scans</p>
                      </div>
                    </div>
                    <div className="stat-item" data-testid="stat-fake">
                      <XCircle className="stat-icon text-red-500" />
                      <div>
                        <p className="stat-value">{stats.fake_detected}</p>
                        <p className="stat-label">Fakes Detected</p>
                      </div>
                    </div>
                    <div className="stat-item" data-testid="stat-real">
                      <CheckCircle className="stat-icon text-green-500" />
                      <div>
                        <p className="stat-value">{stats.real_detected}</p>
                        <p className="stat-label">Real Images</p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* History Card */}
            <Card className="history-card" data-testid="history-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5" />
                  Recent Scans
                </CardTitle>
              </CardHeader>
              <CardContent>
                {history.length === 0 ? (
                  <p className="no-history" data-testid="no-history">No scans yet</p>
                ) : (
                  <div className="history-list">
                    {history.map((item) => (
                      <div key={item.id} className="history-item" data-testid="history-item">
                        <Badge
                          className={item.label === "FAKE" ? "badge-fake-sm" : "badge-real-sm"}
                          data-testid={`history-badge-${item.id}`}
                        >
                          {item.label}
                        </Badge>
                        <span className="history-confidence" data-testid={`history-confidence-${item.id}`}>
                          {item.confidence.toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Model Status */}
            {modelStatus && (
              <Card className="status-card" data-testid="status-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertCircle className="w-5 h-5" />
                    Model Status
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="status-content">
                    <Badge
                      className={modelStatus.loaded ? "badge-success" : "badge-warning"}
                      data-testid="model-status-badge"
                    >
                      {modelStatus.loaded ? "Loaded" : "Not Loaded"}
                    </Badge>
                    {!modelStatus.loaded && (
                      <p className="status-warning" data-testid="status-warning">
                        Run <code>python train.py</code> to train the model
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
