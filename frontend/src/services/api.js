import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Separate instance with long timeout for the analyze call
// (NLI model download can take time on first run)
const apiLong = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes
});

/**
 * Check API health status.
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

/**
 * Submit a literature review.
 * @param {string} literatureReview - The literature review text.
 */
export const submitLiteratureReview = async (literatureReview) => {
  const response = await api.post('/literature-review', {
    literature_review: literatureReview,
  });
  return response.data;
};

/**
 * Upload multiple PDF files.
 * @param {File[]} files - Array of PDF File objects.
 */
export const uploadPapers = async (files) => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });
  const response = await api.post('/papers/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

/**
 * Start analysis pipeline.
 * @param {string} literatureReviewId
 * @param {string[]} paperFilenames
 */
export const startAnalysis = async (literatureReviewId, paperFilenames) => {
  const response = await apiLong.post('/analyze', {
    literature_review_id: literatureReviewId,
    paper_filenames: paperFilenames,
  });
  return response.data;
};

export default api;

/**
 * Verify references in an uploaded paper.
 * @param {string} filename
 */
export const verifyReferences = async (filename) => {
  const response = await apiLong.get(`/papers/${filename}/verify-references`);
  return response.data;
};
