import { useState, useRef, useEffect } from 'react';
import axios from 'axios';

// Use direct backend URL to avoid Next.js proxy for large uploads
const backendURL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const REQUEST_TIMEOUT = 600000; // 10 minutes timeout for very long processing

// Create axios instance with increased timeout
const axiosInstance = axios.create({
  baseURL: backendURL,
  timeout: REQUEST_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Log the backend URL during development
if (process.env.NODE_ENV === 'development') {
  console.log(`API URL: ${backendURL}`);
}

export const useProcessing = () => {
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoadingResult, setIsLoadingResult] = useState(false);
  const [jobId, setJobId] = useState(null);
  
  // References for cleanup
  const progressIntervalRef = useRef(null);
  const pollIntervalRef = useRef(null);
  const cancelTokenSourceRef = useRef(null);

  // Cleanup function for side effects
  useEffect(() => {
    // Reset state on mount
    setProcessing(false);
    setProgress(0);
    setError(null);
    
    return () => {
      // Clear any intervals when component unmounts
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
      
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      
      // Cancel any pending requests when component unmounts
      if (cancelTokenSourceRef.current) {
        cancelTokenSourceRef.current.cancel('Component unmounted');
        cancelTokenSourceRef.current = null;
      }
    };
  }, []);

  // Process the results and set state
  const processResult = (resultData) => {
    try {
      // Format and validate result data
      const formattedResult = formatResult(resultData);
      
      // Set the formatted result
      setResult(formattedResult);
      
      // Check if there's an error in the result
      if (formattedResult.error) {
        setError(`⚠ ${formattedResult.error}`);
      }
      
      // Update progress
      setProgress(100);
      setIsLoadingResult(false);
      setProcessing(false);
    } catch (error) {
      setError(`⚠ Error processing results: ${error.message}`);
      setProcessing(false);
      setIsLoadingResult(false);
    }
  };

  // Poll for job result
  const pollJobStatus = async (jobId) => {
    try {
      setIsLoadingResult(true);
      
      const response = await axiosInstance.get(`/api/results/${jobId}`);
      
      // If status code is 202, it's still processing
      if (response.status === 202) {
        return; // Continue polling
      }
      
      // Stop polling
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      
      // Process results
      processResult(response.data);
      
    } catch (error) {
      if (error.response && error.response.status === 202) {
        // Still processing, continue polling
        return;
      }
      
      // Stop polling on error
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      
      setError(`⚠ Error checking job status: ${error.message}`);
      setProcessing(false);
      setIsLoadingResult(false);
    }
  };

  const handleProcessing = async (endpoint, data, headers = {}) => {
    setProcessing(true);
    setProgress(5);
    setError(null);
    setResult(null);
    
    // Clear any existing intervals and cancel tokens
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
    
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    
    try {
      // Create a cancel token source
      cancelTokenSourceRef.current = axios.CancelToken.source();
      
      // For long-running operations, simulate progress after upload is complete
      let simulatedProgress = 45;
      
      // Fix API endpoint URL to match the backend implementation
      const apiEndpoint = `/api/${endpoint}`;
      
      const response = await axiosInstance.post(apiEndpoint, data, {
        headers,
        onUploadProgress: (progressEvent) => {
          // Calculate upload progress (max 40%)
          const percentCompleted = Math.round((progressEvent.loaded * 40) / progressEvent.total);
          setProgress(5 + percentCompleted);
          
          // Start simulated progress update for processing phase
          if (percentCompleted >= 40 && !progressIntervalRef.current) {
            progressIntervalRef.current = setInterval(() => {
              // Slow down progress as we get higher to avoid hitting 100% too soon
              const increment = simulatedProgress < 70 ? 1 : 0.5;
              simulatedProgress = Math.min(85, simulatedProgress + increment);
              setProgress(simulatedProgress);
              
              // Stop if we reach the max simulated progress
              if (simulatedProgress >= 85) {
                clearInterval(progressIntervalRef.current);
                progressIntervalRef.current = null;
              }
            }, 5000); // Update every 5 seconds
          }
        },
        cancelToken: cancelTokenSourceRef.current.token
      });
      
      // Clear the progress interval since we got a response
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
      
      if (response.status >= 200 && response.status < 300) {
        // Check if the response contains a job_id for polling
        if (response.data.job_id) {
          setJobId(response.data.job_id);
          
          // Start polling for job status every 5 seconds
          pollIntervalRef.current = setInterval(() => {
            pollJobStatus(response.data.job_id);
          }, 5000);
          
          // Store the initial job response in result for display
          setResult({
            job_id: response.data.job_id,
            status: response.data.status || 'processing',
            file_name: response.data.file_name || ''
          });
          
          return response.data;
        } 
        // If there's no job_id, assume the response contains the result directly
        else {
          // Format and validate the result data
          processResult(response.data);
          return response.data;
        }
      } else {
        const errorMessage = response.data?.detail || response.data?.error || 'An unknown error occurred';
        setError(`⚠ ${errorMessage}`);
        setProcessing(false);
        throw new Error(errorMessage);
      }
    } catch (error) {
      // Clear any intervals
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }
      
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      
      let errorMessage = 'An unknown error occurred while processing your request.';
      
      if (error.message === 'canceled' || error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        errorMessage = 'Request was cancelled - processing took too long.';
      } else if (error.response) {
        // Server returned an error response
        errorMessage = error.response.data?.detail || 
                      error.response.data?.error || 
                      error.response.data?.message ||
                      error.message || 
                      'Server error';
      } else if (error.request) {
        // Request was made but no response received
        errorMessage = 'No response received from server. Please check your connection and try again.';
      }
      
      console.error('Processing error:', error);
      setError(`⚠ ${errorMessage}`);
      setProcessing(false);
      throw error;
    }
  };

  // Check and format the result, handling potential errors in component data
  const formatResult = (resultData) => {
    try {
      // Handle null or undefined data
      if (!resultData) {
        return {
          error: "No result data received"
        };
      }
      
      // Handle error responses
      if (resultData.error) {
        return {
          error: resultData.error
        };
      }
      
      // Extract key data from the response
      const {
        transcript,
        segments,
        extractive_summary,
        abstractive_summary,
        topics,
        metadata,
        audio_url,
        job_id,
        status,
        file_name
      } = resultData;
      
      // Format and return the processed result
      return {
        transcript: transcript || "",
        segments: segments || [],
        extractiveSummary: extractive_summary || resultData.extractiveSummary || [],
        abstractiveSummary: abstractive_summary || resultData.abstractiveSummary || "",
        topics: topics || [],
        metadata: metadata || {},
        audioUrl: audio_url || "",
        jobId: job_id || "",
        status: status || "completed",
        fileName: file_name || "",
        error: null
      };
    } catch (error) {
      console.error("Error formatting result:", error);
      return {
        error: `Error processing result: ${error.message}`
      };
    }
  };

  return {
    processing,
    progress,
    result,
    error,
    isLoadingResult,
    handleProcessing,
    setError,
    jobId
  };
}; 