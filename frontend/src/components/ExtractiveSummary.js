"use client";

import React, { useState, useMemo, useCallback } from "react";
import { ListOrdered, AlertTriangle, ChevronRight } from "lucide-react";
import PropTypes from 'prop-types';

// Helper function to process array of summary points
const processSummaryArray = (arr) => {
  const processed = Array.isArray(arr) 
    ? arr.map(item => {
        if (typeof item === 'string') {
          return item;
        }
        if (item && typeof item === 'object') {
          return item.text || item.content || item.sentence || JSON.stringify(item);
        }
        return String(item);
      }).filter(item => 
        item && 
        item.trim() !== "" && 
        !item.startsWith("Error:") && 
        !item.includes("Error generating extractive summary") &&
        !item.includes("timed out") &&
        !item.includes("skipped"))
    : [];
    
  const hasError = Array.isArray(arr) && 
                  arr.some(item => {
                    const text = typeof item === 'string' ? item : 
                                (item && typeof item === 'object') ? 
                                  (item.text || item.content || item.sentence || '') : '';
                    
                    return text && 
                      (text.startsWith("Error:") || 
                       text.includes("Error generating extractive summary") ||
                       text.includes("timed out") ||
                       text.includes("skipped"));
                  });
                   
  return { processedSummary: processed, hasError };
};

const ExtractiveSummary = ({ 
  summary = [],
  maxPoints = 5,
  transcript = ""
}) => {
  const [expanded, setExpanded] = useState(false);

  // Process the summary to ensure it's an array and has items
  const { processedSummary, hasError } = useMemo(() => {
    // Handle if summary is an object with points/items property
    if (summary && typeof summary === 'object' && !Array.isArray(summary)) {
      const summaryArray = summary.points || summary.items || summary.sentences || [];
      return processSummaryArray(summaryArray);
    }
    
    // Handle array case
    return processSummaryArray(summary);
  }, [summary]);

  // If we have no summary but have transcript, create simple key points from transcript
  const finalSummary = useMemo(() => {
    if (processedSummary.length === 0 && transcript && typeof transcript === 'string' && transcript.trim().length > 0) {
      // Split transcript into sentences
      const sentences = transcript.split(/[.!?]+/).filter(s => s.trim().length > 0);
      
      // If we have sentences, use them as key points
      if (sentences.length > 0) {
        // Take up to 3 sentences as key points
        return sentences.slice(0, 3).map(s => s.trim());
      }
    }
    return processedSummary;
  }, [processedSummary, transcript]);
  
  // Get the displayed summary points, limited by maxPoints if not expanded
  const displayedSummary = useMemo(() => 
    expanded ? finalSummary : finalSummary.slice(0, maxPoints),
    [expanded, finalSummary, maxPoints]
  );
  
  const toggleExpanded = useCallback(() => {
    setExpanded(prev => !prev);
  }, []);
  
  if (!finalSummary.length) {
    return (
      <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/40 shadow-lg">
        <div className="flex items-center mb-4">
          <ListOrdered className="mr-3 text-blue-400" size={18} />
          <h2 className="text-lg font-medium text-slate-200">Key Points</h2>
        </div>
        <div className="text-slate-400 text-sm p-3 bg-slate-800/70 rounded-lg border border-slate-700/50">
          {hasError ? (
            <div className="flex items-start">
              <AlertTriangle size={16} className="text-amber-400 mr-2 mt-0.5 flex-shrink-0" />
              <p className="text-amber-300">Unable to generate key points for this content.</p>
            </div>
          ) : (
            <p className="flex items-center">
              <ListOrdered className="mr-2 text-slate-500" size={16} />
              No key points available.
            </p>
          )}
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/40 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <ListOrdered className="mr-3 text-blue-400" size={18} />
          <h2 className="text-lg font-medium text-slate-200">Key Points</h2>
        </div>
        {finalSummary.length > maxPoints && (
          <button 
            onClick={toggleExpanded}
            className="text-xs text-slate-400 hover:text-slate-300 transition-colors"
          >
            {expanded ? 'Show less' : 'Show all'}
          </button>
        )}
      </div>
      
      <div className="space-y-3">
        {displayedSummary.map((item, index) => (
          <div key={`${index}-${typeof item === 'string' ? item.substring(0, 20) : index}`} className="flex items-start gap-3 p-3 bg-slate-800/40 rounded-lg border border-slate-700/30 hover:bg-slate-800/60 transition-colors group">
            <div className="mt-0.5">
              <div className="flex items-center justify-center w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 text-xs font-medium">
                {index + 1}
              </div>
            </div>
            <div className="flex-1">
              <p className="text-slate-300 text-sm leading-relaxed">{item}</p>
            </div>
          </div>
        ))}
        
        {!expanded && finalSummary.length > maxPoints && (
          <button 
            onClick={toggleExpanded}
            className="w-full py-2 flex items-center justify-center gap-1.5 text-sm text-slate-400 hover:text-slate-300 bg-slate-800/40 rounded-lg border border-slate-700/30 hover:bg-slate-800/60 transition-colors"
          >
            <span>Show {finalSummary.length - maxPoints} more points</span>
            <ChevronRight size={14} />
          </button>
        )}
      </div>
    </div>
  );
};

ExtractiveSummary.propTypes = {
  summary: PropTypes.oneOfType([
    PropTypes.array,
    PropTypes.object
  ]),
  maxPoints: PropTypes.number,
  transcript: PropTypes.string
};

export default ExtractiveSummary; 