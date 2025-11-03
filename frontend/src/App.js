// import { useState } from "react";
// import axios from "axios";
// import { Upload, CheckCircle, XCircle } from "lucide-react";
// import { Button } from "@/components/ui/button";
// import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
// import { Progress } from "@/components/ui/progress";
// import { Badge } from "@/components/ui/badge";
// import { toast } from "sonner";
// import "@/App.css";

// // ✅ Backend API URL
// const API_URL = "http://127.0.0.1:8000/predict";

// function App() {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [preview, setPreview] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [result, setResult] = useState(null);

//   // 📁 Handle image selection
//   const handleFileSelect = (event) => {
//     const file = event.target.files[0];
//     if (file) {
//       if (!file.type.startsWith("image/")) {
//         toast.error("Please select a valid image file");
//         return;
//       }
//       setSelectedFile(file);
//       setResult(null);
//       const reader = new FileReader();
//       reader.onloadend = () => setPreview(reader.result);
//       reader.readAsDataURL(file);
//     }
//   };

//   // 🧠 Send image to backend for prediction
//   const handlePredict = async () => {
//     if (!selectedFile) {
//       toast.error("Please select an image first");
//       return;
//     }

//     setLoading(true);
//     const formData = new FormData();
//     formData.append("file", selectedFile);

//     try {
//       const response = await axios.post(API_URL, formData, {
//         headers: { "Content-Type": "multipart/form-data" },
//       });

//       const data = response.data;
//       setResult({
//         label: data.prediction === "Fake" ? "FAKE" : "REAL",
//         confidence: Math.abs(100 - data.score * 100),
//       });

//       toast.success("Analysis complete!");
//     } catch (error) {
//       console.error("Prediction error:", error);
//       toast.error("Prediction failed. Please check backend.");
//     } finally {
//       setLoading(false);
//     }
//   };

//   // 🔁 Reset file
//   const resetAnalysis = () => {
//     setSelectedFile(null);
//     setPreview(null);
//     setResult(null);
//   };

//   return (
//     <div className="app-container">
//       <div className="hero-section">
//         <div className="hero-content">
//           <div className="hero-badge">
//             <span className="badge-dot"></span> AI-Powered Detection
//           </div>
//           <h1 className="hero-title">Deepfake Image Detection</h1>
//           <p className="hero-subtitle">
//             Detect manipulated or fake images using a trained deep learning model.
//           </p>
//         </div>
//       </div>

//       <div className="main-content">
//         <div className="upload-section">
//           <Card className="upload-card">
//             <CardHeader>
//               <CardTitle>Upload Image</CardTitle>
//               <CardDescription>
//                 Select an image to analyze for deepfake detection
//               </CardDescription>
//             </CardHeader>
//             <CardContent>
//               {!preview ? (
//                 <label className="upload-area">
//                   <input
//                     type="file"
//                     accept="image/*"
//                     onChange={handleFileSelect}
//                     className="file-input"
//                   />
//                   <div className="upload-content">
//                     <Upload className="upload-icon" />
//                     <p className="upload-text">Click to upload or drag and drop</p>
//                     <p className="upload-hint">PNG, JPG, JPEG up to 10MB</p>
//                   </div>
//                 </label>
//               ) : (
//                 <div className="preview-container">
//                   <img src={preview} alt="Preview" className="preview-image" />
//                   <Button
//                     onClick={resetAnalysis}
//                     variant="outline"
//                     className="reset-btn"
//                   >
//                     Choose Different Image
//                   </Button>
//                 </div>
//               )}

//               {preview && (
//                 <Button
//                   onClick={handlePredict}
//                   disabled={loading}
//                   className="analyze-btn"
//                 >
//                   {loading ? "Analyzing..." : "Analyze Image"}
//                 </Button>
//               )}
//             </CardContent>
//           </Card>

//           {/* ✅ Show Result */}
//           {result && (
//             <Card className="result-card">
//               <CardHeader>
//                 <CardTitle className="flex items-center gap-2">
//                   {result.label === "FAKE" ? (
//                     <XCircle className="text-red-500" />
//                   ) : (
//                     <CheckCircle className="text-green-500" />
//                   )}
//                   Detection Result
//                 </CardTitle>
//               </CardHeader>
//               <CardContent>
//                 <div className="result-content">
//                   <div className="result-label-section">
//                     <Badge
//                       className={
//                         result.label === "FAKE" ? "badge-fake" : "badge-real"
//                       }
//                     >
//                       {result.label}
//                     </Badge>
//                     {/* <p className="confidence-text">
//                       {result.confidence.toFixed(2)}% Confidence
//                     </p> */}
//                   </div>
//                   <Progress
//                     value={result.confidence}
//                     className="confidence-progress"
//                   />
//                   <p className="result-description">
//                     {result.label === "FAKE"
//                       ? "This image shows signs of manipulation and is likely a deepfake."
//                       : "This image appears to be authentic with no signs of manipulation."}
//                   </p>
//                 </div>
//               </CardContent>
//             </Card>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }

// export default App;






// // import { useState } from "react";
// // import axios from "axios";
// // import { Upload, CheckCircle, XCircle } from "lucide-react";
// // import { Button } from "@/components/ui/button";
// // import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
// // import { Progress } from "@/components/ui/progress";
// // import { Badge } from "@/components/ui/badge";
// // import { toast } from "sonner";
// // import "@/App.css";

// // // Backend API URL
// // const API_URL = "http://127.0.0.1:8000/predict";

// // function App() {
// //   const [selectedFile, setSelectedFile] = useState(null);
// //   const [preview, setPreview] = useState(null);
// //   const [loading, setLoading] = useState(false);
// //   const [result, setResult] = useState(null);

// //   const handleFileSelect = (event) => {
// //     const file = event.target.files[0];
// //     if (file) {
// //       if (!file.type.startsWith("image/")) {
// //         toast.error("Please select a valid image file");
// //         return;
// //       }
// //       setSelectedFile(file);
// //       setResult(null);
// //       const reader = new FileReader();
// //       reader.onloadend = () => setPreview(reader.result);
// //       reader.readAsDataURL(file);
// //     }
// //   };

// //   const handlePredict = async () => {
// //     if (!selectedFile) {
// //       toast.error("Please select an image first");
// //       return;
// //     }

// //     setLoading(true);
// //     const formData = new FormData();
// //     formData.append("file", selectedFile);

// //     try {
// //       const response = await axios.post(API_URL, formData, {
// //         headers: { "Content-Type": "multipart/form-data" },
// //       });

// //       const data = response.data;
// //       console.log("Backend response:", data);

// //       // ----- Robust parsing -----
// //       // backend might return { result: "FAKE", confidence: 82.5 }
// //       // or { prediction: "Fake", score: 0.825 }
// //       // or { score: 0.825 } (only score) etc.

// //       // 1) Try to read numeric score/confidence
// //       let rawScore = null;
// //       if (data.score !== undefined) rawScore = data.score;
// //       else if (data.confidence !== undefined) rawScore = data.confidence;
// //       // if rawScore looks like percentage (e.g., 82.5), normalize to 0-1
// //       if (rawScore !== null) {
// //         rawScore = Number(rawScore);
// //         if (rawScore > 1) rawScore = rawScore / 100.0; // convert percent -> 0-1
// //       }

// //       // 2) Determine label string if backend gave it directly
// //       let labelStr = null;
// //       if (data.result) labelStr = String(data.result);
// //       else if (data.prediction) labelStr = String(data.prediction);

// //       // 3) If we don't have a label string, derive from score
// //       if (!labelStr && rawScore !== null) {
// //         labelStr = rawScore >= 0.5 ? "Fake" : "Real";
// //       }

// //       // 4) Final normalized outputs
// //       const label = (labelStr || "Real").toUpperCase() === "FAKE" || (labelStr || "").toLowerCase() === "fake" ? "FAKE" : "REAL";
// //       let confidencePercent = 0;
// //       if (rawScore !== null) {
// //         confidencePercent = Number((rawScore * 100).toFixed(2));
// //       } else if (data.confidence !== undefined) {
// //         // fallback if confidence was already percent
// //         confidencePercent = Number(data.confidence);
// //       } else {
// //         confidencePercent = 0;
// //       }

// //       setResult({
// //         label,
// //         confidence: confidencePercent,
// //       });

// //       toast.success("Analysis complete!");
// //     } catch (error) {
// //       console.error("Prediction error:", error);
// //       toast.error("Prediction failed. Please check backend.");
// //     } finally {
// //       setLoading(false);
// //     }
// //   };

// //   const resetAnalysis = () => {
// //     setSelectedFile(null);
// //     setPreview(null);
// //     setResult(null);
// //   };

// //   return (
// //     <div className="app-container">
// //       <div className="hero-section">
// //         <div className="hero-content">
// //           <div className="hero-badge">
// //             <span className="badge-dot"></span> AI-Powered Detection
// //           </div>
// //           <h1 className="hero-title">Deepfake Image Detection</h1>
// //           <p className="hero-subtitle">
// //             Detect manipulated or fake images using a trained deep learning model.
// //           </p>
// //         </div>
// //       </div>

// //       <div className="main-content">
// //         <div className="upload-section">
// //           <Card className="upload-card">
// //             <CardHeader>
// //               <CardTitle>Upload Image</CardTitle>
// //               <CardDescription>Select an image to analyze for deepfake detection</CardDescription>
// //             </CardHeader>
// //             <CardContent>
// //               {!preview ? (
// //                 <label className="upload-area">
// //                   <input type="file" accept="image/*" onChange={handleFileSelect} className="file-input" />
// //                   <div className="upload-content">
// //                     <Upload className="upload-icon" />
// //                     <p className="upload-text">Click to upload or drag and drop</p>
// //                     <p className="upload-hint">PNG, JPG, JPEG up to 10MB</p>
// //                   </div>
// //                 </label>
// //               ) : (
// //                 <div className="preview-container">
// //                   <img src={preview} alt="Preview" className="preview-image" />
// //                   <Button onClick={resetAnalysis} variant="outline" className="reset-btn">Choose Different Image</Button>
// //                 </div>
// //               )}

// //               {preview && (
// //                 <Button onClick={handlePredict} disabled={loading} className="analyze-btn">
// //                   {loading ? "Analyzing..." : "Analyze Image"}
// //                 </Button>
// //               )}
// //             </CardContent>
// //           </Card>

// //           {result && (
// //             <Card className="result-card">
// //               <CardHeader>
// //                 <CardTitle className="flex items-center gap-2">
// //                   {result.label === "FAKE" ? <XCircle className="text-red-500" /> : <CheckCircle className="text-green-500" />}
// //                   Detection Result
// //                 </CardTitle>
// //               </CardHeader>
// //               <CardContent>
// //                 <div className="result-content">
// //                   <div className="result-label-section">
// //                     <Badge className={result.label === "FAKE" ? "badge-fake" : "badge-real"}>{result.label}</Badge>
// //                     <p className="confidence-text">Confidence: {result.confidence.toFixed(2)}%</p>
// //                   </div>
// //                   <Progress value={result.confidence} className="confidence-progress" />
// //                   <p className="result-description">
// //                     {result.label === "FAKE"
// //                       ? "This image shows signs of manipulation and is likely a deepfake."
// //                       : "This image appears to be authentic with no signs of manipulation."}
// //                   </p>
// //                 </div>
// //               </CardContent>
// //             </Card>
// //           )}
// //         </div>
// //       </div>
// //     </div>
// //   );
// // }

// // export default App;






import { useState } from "react";
import axios from "axios";
import { Upload, CheckCircle, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import "@/App.css";

// ✅ Backend API URL
const API_URL = "http://127.0.0.1:8000/predict";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // 📁 Handle image selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        toast.error("Please select a valid image file");
        return;
      }
      setSelectedFile(file);
      setResult(null);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  // 🧠 Send image to backend for prediction
  const handlePredict = async () => {
    if (!selectedFile) {
      toast.error("Please select an image first");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(API_URL, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = response.data;
      console.log("✅ Backend response:", data);

      // ✅ Adjust based on backend keys
      const label =
        (data.result || "").toUpperCase() === "FAKE" ? "FAKE" : "REAL";
      const confidence = Number(data.confidence || 0);

      setResult({ label, confidence });
      toast.success("Analysis complete!");
    } catch (error) {
      console.error("❌ Prediction error:", error);
      toast.error("Prediction failed. Please check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  // 🔁 Reset file
  const resetAnalysis = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="app-container">
      {/* 🔷 Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <div className="hero-badge">
            <span className="badge-dot"></span> AI-Powered Detection
          </div>
          <h1 className="hero-title">Deepfake Image Detection</h1>
          <p className="hero-subtitle">
            Detect manipulated or fake images using a trained deep learning
            model.
          </p>
        </div>
      </div>

      {/* 🖼️ Upload Section */}
      <div className="main-content">
        <div className="upload-section">
          <Card className="upload-card">
            <CardHeader>
              <CardTitle>Upload Image</CardTitle>
              <CardDescription>
                Select an image to analyze for deepfake detection
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!preview ? (
                <label className="upload-area">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="file-input"
                  />
                  <div className="upload-content">
                    <Upload className="upload-icon" />
                    <p className="upload-text">
                      Click to upload or drag and drop
                    </p>
                    <p className="upload-hint">PNG, JPG, JPEG up to 10MB</p>
                  </div>
                </label>
              ) : (
                <div className="preview-container">
                  <img src={preview} alt="Preview" className="preview-image" />
                  <Button
                    onClick={resetAnalysis}
                    variant="outline"
                    className="reset-btn"
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
                >
                  {loading ? "Analyzing..." : "Analyze Image"}
                </Button>
              )}
            </CardContent>
          </Card>

          {/* ✅ Show Result */}
          {result && (
            <Card className="result-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {result.label === "FAKE" ? (
                    <XCircle className="text-red-500" />
                  ) : (
                    <CheckCircle className="text-green-500" />
                  )}
                  Detection Result
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="result-content">
                  <div className="result-label-section">
                    <Badge
                      className={
                        result.label === "FAKE" ? "badge-fake" : "badge-real"
                      }
                    >
                      {result.label}
                    </Badge>
                    <p className="confidence-text">
                      Confidence: {result.confidence.toFixed(2)}%
                    </p>
                  </div>
                  <Progress
                    value={result.confidence}
                    className="confidence-progress"
                  />
                  <p className="result-description">
                    {result.label === "FAKE"
                      ? "⚠️ This image shows signs of manipulation and is likely a deepfake."
                      : "✅ This image appears authentic with no signs of manipulation."}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
