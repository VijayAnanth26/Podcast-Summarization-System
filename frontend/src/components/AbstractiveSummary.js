"use client";

import React, { useMemo } from "react";
import { Layers, AlertTriangle } from "lucide-react";
import PropTypes from 'prop-types';

// Helper function to process summary
const processSummary = (summary) => {
  // Handle null or undefined
  if (!summary) {
    return { text: '', model: '', hasError: false };
  }
  
  // Handle array of strings
  if (Array.isArray(summary)) {
    const filtered = summary
      .filter(s => s && typeof s === 'string' && s.trim() !== '')
      .map(s => s.trim());
    
    return {
      text: filtered.join('\n\n'),
      model: summary.model || '',
      hasError: filtered.some(s => 
        s.startsWith('Error:') || 
        s.includes('Error generating abstractive summary') ||
        s.includes('timed out') ||
        s.includes('skipped')
      )
    };
  }
  
  // Handle string
  if (typeof summary === 'string') {
    const trimmedText = summary.trim();
    return {
      text: trimmedText,
      model: '',
      hasError: trimmedText.startsWith('Error:') || 
                trimmedText.includes('Error generating abstractive summary') ||
                trimmedText.includes('timed out') ||
                trimmedText.includes('skipped')
    };
  }
  
  // Handle object
  if (summary && typeof summary === 'object') {
    // Try to extract text from various possible properties
    const text = summary.text || summary.content || summary.summary || '';
    const trimmedText = typeof text === 'string' ? text.trim() : '';
    
    return {
      text: trimmedText,
      model: summary.model || summary.model_name || summary.modelName || '',
      hasError: trimmedText.startsWith('Error:') || 
                trimmedText.includes('Error generating abstractive summary') ||
                trimmedText.includes('timed out') ||
                trimmedText.includes('skipped')
    };
  }
  
  return { text: '', model: '', hasError: false };
};

const AbstractiveSummary = ({ 
  summary = {
    text: '',
    confidence: 0,
    model: ''
  },
  transcript = ''
}) => {
  // Process summary data
  const { text, model, hasError } = useMemo(() => 
    processSummary(summary),
    [summary]
  );
  
  // Check if text is empty or too short to be a real summary
  const isEmpty = !text || text.length < 5 || text === "No abstractive summary available.";
  
  // If we have a transcript but no summary, create a simple summary
  const finalText = useMemo(() => {
    if ((isEmpty || hasError) && transcript && transcript.length > 0) {
      // Get the first sentence of the transcript as a simple summary
      const firstSentence = transcript.split(/[.!?]+/)[0].trim();
      if (firstSentence && firstSentence.length > 5) {
        return firstSentence;
      }
    }
    return text;
  }, [isEmpty, hasError, transcript, text]);
  
  // Check if we still have no valid summary
  const hasNoSummary = !finalText || finalText.length < 5;
  
  if (hasNoSummary || hasError) {
    return (
      <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/40 shadow-lg">
        <div className="flex items-center mb-4">
          <Layers className="mr-3 text-blue-400" size={18} />
          <h2 className="text-lg font-medium text-slate-200">Abstractive Summary</h2>
        </div>
        <div className="text-slate-400 text-sm p-3 bg-slate-800/70 rounded-lg border border-slate-700/50">
          {hasError ? (
            <div className="flex items-start">
              <AlertTriangle size={16} className="text-amber-400 mr-2 mt-0.5 flex-shrink-0" />
              <p className="text-amber-300">Unable to generate summary for this content.</p>
            </div>
          ) : (
            <p>The content is too short for a meaningful summary.</p>
          )}
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/40 shadow-lg">
      <div className="flex items-center mb-4">
        <Layers className="mr-3 text-blue-400" size={18} />
        <h2 className="text-lg font-medium text-slate-200">Abstractive Summary</h2>
      </div>
      
      <div className="prose prose-sm prose-invert max-w-none">
        <p className="text-slate-300 leading-relaxed">
          {finalText}
        </p>
      </div>
      
      {model && (
        <div className="mt-4 flex items-center justify-start gap-2">
          <span className="text-xs text-slate-500">Generated with {model}</span>
        </div>
      )}
    </div>
  );
};

AbstractiveSummary.propTypes = {
  summary: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.arrayOf(PropTypes.string),
    PropTypes.shape({
      text: PropTypes.string,
      confidence: PropTypes.number,
      model: PropTypes.string
    })
  ]),
  transcript: PropTypes.string
};

export default AbstractiveSummary; 