import { useEffect, useState } from "react";
import "./App.css";
import {
  databricksSetup,
  uploadAndAnalyzePdf,
} from "./services/databricksService";

interface UploadedFile {
  lastModified?: number;
  lastModifiedDate?: Date;
  name?: string;
  size?: number;
  type?: string;
  webkitRelativePath?: string;
}
interface AiAnalysisResponse {
  responses: {
    prompt: string;
    answer: string;
    success: boolean;
    error: string | null;
  }[];
  merged_summary: string;
}

function App() {
  const [loading, setLoading] = useState<boolean>(true);
  const [databricksConnected, setDatabricksConnected] =
    useState<boolean>(false);
  const [processingFileStatus, setProcessingFileStatus] = useState<
    string | null
  >(null);
  const [aiAnalysis, setAiAnalysis] = useState<AiAnalysisResponse | null>(null);
  useEffect(() => {
    initializeApp();
  }, []);
  const initializeApp = async () => {
    try {
      setLoading(true);

      console.log("Attempting auto-connection to Databricks...");
      const connectResult = await databricksSetup();
      console.log({ connectResult });

      if (connectResult?.success) {
        console.log("Auto-connection successful");
        setDatabricksConnected(true);
      }

      setLoading(false);
    } catch (error) {
      console.error("Failed to initialize app:", error);
      setDatabricksConnected(false);
      setLoading(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.includes("pdf")) {
      console.log("Please select a PDF file");
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      console.log("File size must be less than 50MB");
      return;
    }
    await processFile(file);
  };
  const processFile = async (file: UploadedFile) => {
    try {
      setProcessingFileStatus("Uploading and analyzing File...");

      const result = await uploadAndAnalyzePdf(file, true);
      console.log({ result });
      if (result?.success) {
        setAiAnalysis(result?.analysis);
      }

      setProcessingFileStatus(null);
    } catch (error) {
      console.error("Failed to initialize app:", error);
      setDatabricksConnected(false);
      setProcessingFileStatus(null);
    }
  };
  return (
    <div className="app-container">
      {loading ? (
        <div className="status">Loading...</div>
      ) : (
        <div className="status">
          <span
            className={`led ${databricksConnected ? "led-green" : "led-red"}`}
          ></span>
          {databricksConnected ? "Connected" : "Disconnected"}
          {!databricksConnected && (
            <button className="retry-btn" onClick={initializeApp}>
              Retry Connection
            </button>
          )}
        </div>
      )}
      <div className="upload-box">
        <span>
          {processingFileStatus ? processingFileStatus : "Upload a PDF"}
        </span>
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          disabled={
            loading || !databricksConnected || Boolean(processingFileStatus)
          }
        />
      </div>

      {aiAnalysis && (
        <div className="analysis-results">
          <h3>AI Analysis Results</h3>

          <div className="individual-responses">
            {aiAnalysis.responses.map((res, idx) => (
              <div key={idx} className="response-card">
                <h4>Prompt {idx + 1}</h4>
                <p>
                  <strong>Prompt:</strong> {res.prompt}
                </p>
                <p>
                  <strong>Answer:</strong>
                </p>
                <div className="answer-box">{res.answer}</div>
              </div>
            ))}
          </div>

          <div className="merged-response">
            <h3>Merged Summary</h3>
            <div className="answer-box">{aiAnalysis.merged_summary}</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
