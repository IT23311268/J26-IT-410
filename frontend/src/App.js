import React, { useState, useCallback } from 'react';
import './App.css';
import LiteratureReviewInput from './components/LiteratureReviewInput';
import PdfUploader from './components/PdfUploader';
import AnalyzeButton from './components/AnalyzeButton';
import ValidationMessage from './components/ValidationMessage';
import AnalysisResults from './components/AnalysisResults';
import ReferenceVerification from './components/ReferenceVerification';
import { submitLiteratureReview, uploadPapers, startAnalysis } from './services/api';

function App() {
  // ── State ──
  const [literatureReview, setLiteratureReview] = useState('');
  const [pdfFiles, setPdfFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [uploadedPaperFilenames, setUploadedPaperFilenames] = useState([]);

  // ── Handlers ──
  const handleAddFiles = useCallback((newFiles) => {
    setPdfFiles((prev) => {
      const existingNames = new Set(prev.map((f) => f.name));
      const unique = newFiles.filter((f) => !existingNames.has(f.name));
      return [...prev, ...unique];
    });
  }, []);

  const handleRemoveFile = useCallback((index) => {
    setPdfFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleAnalyze = async () => {
    // Clear previous messages
    setError(null);
    setSuccessMessage(null);
    setAnalysisResult(null);

    // ── Validation ──
    if (!literatureReview.trim()) {
      setError('Please paste your Literature Review text before analyzing.');
      return;
    }
    if (literatureReview.trim().length < 50) {
      setError('Literature Review is too short. Please provide at least 50 characters.');
      return;
    }
    if (pdfFiles.length === 0) {
      setError('Please upload at least one reference paper (PDF) before analyzing.');
      return;
    }

    setIsLoading(true);

    try {
      // Step 1: Submit literature review
      const reviewResponse = await submitLiteratureReview(literatureReview);

      // Step 2: Upload PDFs
      const uploadResponse = await uploadPapers(pdfFiles);

      // Step 3: Start analysis
      const paperFilenames = uploadResponse.papers.map((p) => p.saved_filename);
      setUploadedPaperFilenames(paperFilenames);
      const analysisResponse = await startAnalysis(reviewResponse.id, paperFilenames);

      setSuccessMessage(
        `Literature review submitted (${reviewResponse.character_count} chars) and ${uploadResponse.total_uploaded} paper(s) uploaded successfully.`
      );
      setAnalysisResult(analysisResponse);
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.message ||
        'An unexpected error occurred. Please try again.';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>Claim-Level Hallucination Detection and Citation Verification</h1>
        <p>Hallucination Detection & Citation Verification</p>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {/* Validation / Success Messages */}
        <ValidationMessage type="error" message={error} />
        <ValidationMessage type="success" message={successMessage} />

        {/* Literature Review Input */}
        <LiteratureReviewInput
          value={literatureReview}
          onChange={setLiteratureReview}
          disabled={isLoading}
        />

        {/* PDF Upload */}
        <PdfUploader
          files={pdfFiles}
          onAddFiles={handleAddFiles}
          onRemoveFile={handleRemoveFile}
          disabled={isLoading}
        />

        {/* Analyze Button */}
        <AnalyzeButton
          onClick={handleAnalyze}
          isLoading={isLoading}
          disabled={isLoading}
        />

        {/* Results */}
        <AnalysisResults result={analysisResult} />

        {/* Bibliography Audit — always visible; has its own PDF upload */}
        <ReferenceVerification paperFilenames={uploadedPaperFilenames} />
      </main>

      {/* Footer */}
      <footer className="app-footer">
        Hallucination Detection & Citation Verification System © 2026
      </footer>
    </div>
  );
}

export default App;
