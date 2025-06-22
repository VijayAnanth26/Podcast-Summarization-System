import { useState, useCallback } from 'react';
import axios from 'axios';

// Use direct backend URL to avoid Next.js proxy
const backendURL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const REQUEST_TIMEOUT = 120000; // 2 minutes timeout for trimming

// Create axios instance with increased timeout
const axiosInstance = axios.create({
  baseURL: backendURL,
  timeout: REQUEST_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  }
});

export const useTrimming = () => {
  const [trimming, setTrimming] = useState(false);
  const [trimmedClips, setTrimmedClips] = useState([]);
  const [error, setError] = useState(null);
  
  // Function to trim audio/video
  const trimAudio = useCallback(async (jobId, startTime, endTime, highlightText) => {
    if (!jobId) {
      setError("Job ID is required for trimming");
      return null;
    }
    
    if (startTime >= endTime) {
      setError("Start time must be less than end time");
      return null;
    }
    
    setTrimming(true);
    setError(null);
    
    try {
      const response = await axiosInstance.post('/api/trim', {
        job_id: jobId,
        start_time: startTime,
        end_time: endTime,
        highlight_text: highlightText || ""
      });
      
      if (response.status === 200) {
        const newClip = {
          ...response.data,
          createdAt: new Date().toISOString()
        };
        
        // Add to trimmed clips list
        setTrimmedClips(prev => [newClip, ...prev]);
        
        return newClip;
      } else {
        throw new Error("Failed to trim audio/video");
      }
    } catch (err) {
      const errorMessage = err.response?.data?.detail || 
                          err.response?.data?.error || 
                          err.message || 
                          "An unknown error occurred while trimming";
      
      setError(`Error: ${errorMessage}`);
      console.error("Trimming error:", err);
      return null;
    } finally {
      setTrimming(false);
    }
  }, []);
  
  // Function to clear all trimmed clips
  const clearTrimmedClips = useCallback(() => {
    setTrimmedClips([]);
  }, []);
  
  // Function to remove a specific trimmed clip
  const removeTrimmedClip = useCallback((trimId) => {
    setTrimmedClips(prev => prev.filter(clip => clip.trim_id !== trimId));
  }, []);
  
  return {
    trimming,
    trimmedClips,
    error,
    trimAudio,
    clearTrimmedClips,
    removeTrimmedClip
  };
}; 